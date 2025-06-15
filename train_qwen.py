# train_qwen.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging
import os
import json
from datetime import datetime
from modelscope import snapshot_download, AutoModel, AutoTokenizer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 验证标签范围
        if not all(isinstance(label, (int, np.integer)) for label in labels):
            raise ValueError("All labels must be integers")
        if not all(0 <= label < 1000 for label in labels):  # 假设类别数不超过1000
            raise ValueError("Labels must be non-negative integers less than 1000")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])  # 确保标签是整数
        
        # Create prompt template
        prompt = f"Please classify this automotive news article into the appropriate category. Return only the category ID.\nNews text: {text}\nCategory:"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # 创建与输入相同长度的标签序列
        labels = torch.full_like(input_ids, -100)  # 使用-100作为忽略索引
        labels[-1] = label  # 只在最后一个位置设置实际的标签
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    



def load_and_preprocess_data(csv_path, test_size=0.2, val_size=0.1):
    """Load and preprocess the news dataset"""
    logger.info("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # 打印原始数据大小
    logger.info(f"Total number of samples in dataset: {len(df)}")
    
    # Combine title and text for input
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # 打印原始标签的分布
    logger.info("Original label distribution:")
    logger.info(df['label_id'].value_counts().sort_index())
    
    # Convert label_id to numeric and ensure it's within valid range
    df['label_id'] = pd.to_numeric(df['label_id'], errors='coerce')
    
    # 验证标签数据
    if df['label_id'].isna().any():
        logger.warning("Found NaN values in labels, removing these rows")
        df = df.dropna(subset=['label_id'])
    
    if not df['label_id'].apply(lambda x: isinstance(x, (int, np.integer))).all():
        logger.warning("Converting labels to integers")
        df['label_id'] = df['label_id'].astype(int)
    
    # 打印处理后的标签分布
    logger.info("Processed label distribution:")
    logger.info(df['label_id'].value_counts().sort_index())
    
    # 如果有负值标签，将其映射到非负值
    if (df['label_id'] < 0).any():
        logger.warning("Found negative label values, mapping to non-negative values")
        label_map = {label: idx for idx, label in enumerate(sorted(df['label_id'].unique()))}
        df['label_id'] = df['label_id'].map(label_map)
        logger.info("New label mapping:")
        for old_label, new_label in label_map.items():
            logger.info(f"{old_label} -> {new_label}")
    
    texts = df['full_text'].values
    labels = df['label_id'].values
    
    # Split into train, validation and test sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=test_size + val_size, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=test_size/(test_size + val_size), random_state=42
    )
    
    # 打印各个集合的大小
    logger.info(f"Training set size: {len(train_texts)}")
    logger.info(f"Validation set size: {len(val_texts)}")
    logger.info(f"Test set size: {len(test_texts)}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def save_results(results, output_dir):
    """Save experiment results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
    
    # Convert numpy values to Python native types
    results_dict = {
        str(rank): {
            'precision': float(metrics['eval_precision']),
            'recall': float(metrics['eval_recall']),
            'f1': float(metrics['eval_f1']),
            'accuracy': float(metrics['eval_accuracy'])
        }
        for rank, metrics in results.items()
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")
    return results_file



def compute_metrics(eval_preds):
    """Compute metrics for evaluation"""
    predictions, labels = eval_preds
    
    # 获取最后一个位置的预测（因为我们在最后一个位置设置了标签）
    predictions = predictions[:, -1, :]  # shape: [batch_size, vocab_size]
    predictions = np.argmax(predictions, axis=-1)
    
    # 获取最后一个位置的实际标签
    labels = labels[:, -1]  # shape: [batch_size]
    
    # 移除填充标签（-100）
    valid_mask = labels != -100
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    # Calculate metrics
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def train_qwen_model(
    model_name="",
    csv_path="",
    output_dir="",
    lora_rank=8,
    batch_size=4,
    num_epochs=3,
    learning_rate=2e-5,
    max_length=512
):
    """Train Qwen model with LoRA fine-tuning"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model from ModelScope
    logger.info(f"Loading model {model_name} from ModelScope...")
    model_dir = snapshot_download(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.float16
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Load and preprocess data
    (train_texts, train_labels), (val_texts, val_labels), _ = load_and_preprocess_data(csv_path)
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=learning_rate,
        fp16=True,
        no_cuda=False,
        local_rank=-1,
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        save_total_limit=2,
        remove_unused_columns=False
    )
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Clearing cache...")
    # 清理显存
    torch.cuda.empty_cache()
    logger.info("Cache cleared")

    # Evaluate final model
    logger.info("Evaluating final model...")
    final_metrics = trainer.evaluate()
    
    # Print final metrics
    logger.info("Final metrics:")
    for metric_name, value in final_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    return model, tokenizer, final_metrics

def evaluate_trained_model(
    model_path,
    csv_path="dataset/news.csv",
    batch_size=1,
    max_length=512,
    test_size=100  # 只使用部分测试数据
):
    """Evaluate a trained Qwen model using a subset of test data"""
    # Find the latest checkpoint in the rank directory
    checkpoints = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_path}")
    
    # Sort checkpoints by number and get the latest one
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    
    logger.info(f"Loading trained model from {checkpoint_path}...")
    
    # Load tokenizer and model with memory optimizations
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        use_cache=False,  # Disable KV cache
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Load test data
    _, _, (test_texts, test_labels) = load_and_preprocess_data(csv_path)
    
    # 只使用部分测试数据
    if len(test_texts) > test_size:
        logger.info(f"Using {test_size} samples out of {len(test_texts)} test samples")
        # 随机选择测试样本
        indices = np.random.choice(len(test_texts), test_size, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    # Create test dataset
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Create trainer for evaluation with memory optimizations
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="evaluation_results",
            per_device_eval_batch_size=batch_size,
            fp16=True,
            no_cuda=False,
            local_rank=-1,
            dataloader_num_workers=1,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            optim="adamw_torch_fused"
        ),
        compute_metrics=compute_metrics
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        metrics = trainer.evaluate(test_dataset)
        
        # Print metrics
        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error: {e}")
        logger.info("Try reducing test_size or batch_size further")
        raise
    finally:
        # Clear CUDA cache
        torch.cuda.empty_cache()



def run_experiments():
    """Run experiments with different LoRA ranks"""
    ranks = [4, 16, 32]
    results = {}
    
    # Create main output directory
    main_output_dir = "experiment_results"
    os.makedirs(main_output_dir, exist_ok=True)
    
    for rank in ranks:
        logger.info(f"\nTraining with LoRA rank {rank}")
        output_dir = os.path.join(main_output_dir, f"rank_{rank}")
        model, tokenizer, metrics = train_qwen_model(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            csv_path="dataset/news.csv",
            output_dir=output_dir,
            lora_rank=rank,
            batch_size=4,
            num_epochs=1
        )
        results[rank] = metrics
    
    # Save results
    results_file = save_results(results, main_output_dir)
    
    # Print results in table format
    print("\nResults Summary:")
    print("Rank\tPrecision\tRecall\tF1\tAccuracy")
    print("-" * 50)
    for rank, metrics in results.items():
        print(f"{rank}\t{metrics['eval_precision']:.4f}\t{metrics['eval_recall']:.4f}\t{metrics['eval_f1']:.4f}\t{metrics['eval_accuracy']:.4f}")
    
    return results_file


def get_user_input():
    """Get user input for mode and ranks"""
    print("\n请选择操作模式：")
    print("1. 训练模型")
    print("2. 评估模型")
    
    while True:
        mode = input("请输入选项 (1 或 2): ").strip()
        if mode in ['1', '2']:
            break
        print("无效输入，请重新输入")
    
    print("\n请输入要使用的rank值（用空格分隔多个值，例如：4 8 16 32）")
    while True:
        try:
            ranks = [int(x) for x in input("请输入rank值: ").strip().split()]
            if all(r > 0 for r in ranks):
                break
            print("rank值必须为正整数，请重新输入")
        except ValueError:
            print("输入格式错误，请重新输入")
    
    return mode, ranks



if __name__ == "__main__":
    mode, ranks = get_user_input()
    results = {}
    
    if mode == '1':  # 训练模式
        # Create main output directory
        main_output_dir = "experiment_results"
        os.makedirs(main_output_dir, exist_ok=True)
        
        for rank in ranks:
            logger.info(f"\nTraining with LoRA rank {rank}")
            output_dir = os.path.join(main_output_dir, f"rank_{rank}")
            model, tokenizer, metrics = train_qwen_model(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                csv_path="dataset/news.csv",
                output_dir=output_dir,
                lora_rank=rank,
                batch_size=4,
                num_epochs=1
            )
            results[rank] = metrics
        
        # Save results
        results_file = save_results(results, main_output_dir)
        
    else:  # 评估模式
        for rank in ranks:
            model_path = f"experiment_results/rank_{rank}"
            logger.info(f"\nEvaluating model with rank {rank}")
            # 或者指定更小的测试集大小
            metrics = evaluate_trained_model(model_path, test_size=20, batch_size=1) 
            results[rank] = metrics
    
    # 打印结果表格
    print("\nResults Summary:")
    print("Rank\tPrecision\tRecall\tF1\tAccuracy")
    print("-" * 50)
    for rank, metrics in results.items():
        print(f"{rank}\t{metrics['eval_precision']:.4f}\t{metrics['eval_recall']:.4f}\t{metrics['eval_f1']:.4f}\t{metrics['eval_accuracy']:.4f}")