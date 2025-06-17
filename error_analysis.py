# error_analysis.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# 设置日志
def setup_logging():
    """设置日志配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

class NewsDataset(Dataset):
    """新闻数据集类，支持灵活的标签映射和提示模板"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 max_length: int = 512, prompt_template: Optional[str] = None):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or (
            "Please classify this automotive news article into the appropriate category. "
            "Return only the category ID.\nNews text: {text}\nCategory:"
        )
        
        # 验证输入
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        # 验证和处理标签
        self._process_labels(labels)
        
    def _process_labels(self, labels: List[int]):
        """处理和验证标签"""
        # 确保所有标签都是整数
        processed_labels = []
        for label in labels:
            if pd.isna(label):
                raise ValueError("Found NaN values in labels")
            processed_labels.append(int(label))
        
        # 获取唯一标签并创建映射
        unique_labels = sorted(set(processed_labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        
        # 转换标签为连续的非负整数
        self.labels = np.array([self.label_map[label] for label in processed_labels])
        
        logger.info(f"Number of unique labels: {len(unique_labels)}")
        logger.info(f"Label mapping: {self.label_map}")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = int(self.labels[idx])
        
        # 创建提示
        prompt = self.prompt_template.format(text=text)
        
        # 分词
        try:
            encodings = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Error tokenizing text at index {idx}: {e}")
            raise
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # 创建标签序列
        labels = torch.full_like(input_ids, -100)
        labels[-1] = label
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,
            "true_label": label,
            "original_label": self.reverse_label_map[label]
        }

def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """
    加载训练好的模型和分词器
    
    Args:
        model_path: 模型路径
        
    Returns:
        tuple: (model, tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",  # 自动选择设备
            torch_dtype=torch.float16
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_confusion_matrix(model, tokenizer, test_dataset: NewsDataset, 
                            output_dir: str, batch_size: int = 32) -> Dict[str, Any]:
    """
    生成混淆矩阵和详细指标
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        test_dataset: 测试数据集
        output_dir: 输出目录
        batch_size: 批次大小
        
    Returns:
        dict: 包含各种指标的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating confusion matrix for {len(test_dataset)} samples...")

    model.eval()
    all_predictions = []
    all_true_labels = []
    failed_samples = []
    
    # 创建DataLoader
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
            try:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,  # 确定性生成
                    temperature=1.0
                )
                
                predictions = outputs[:, -1].cpu().numpy()
                true_labels = batch["true_label"].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                failed_samples.append(batch_idx)
                continue
    
    if failed_samples:
        logger.warning(f"Failed to process {len(failed_samples)} batches")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # 过滤无效预测
    valid_mask = (all_predictions >= 0) & (all_predictions < len(test_dataset.label_map))
    if not valid_mask.all():
        logger.warning(f"Found {(~valid_mask).sum()} invalid predictions, filtering them out")
        all_predictions = all_predictions[valid_mask]
        all_true_labels = all_true_labels[valid_mask]
    
    # 创建混淆矩阵
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # 计算指标
    metrics = _calculate_metrics(all_true_labels, all_predictions)
    
    # 保存指标和可视化
    _save_metrics(metrics, output_dir)
    _plot_confusion_matrix(cm, output_dir, test_dataset.reverse_label_map)
    
    # 返回结果
    result = {
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'failed_samples': failed_samples,
        **metrics
    }
    
    return result

def _calculate_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """计算各种分类指标"""
    # 总体准确率
    accuracy = accuracy_score(true_labels, predictions)
    
    # 每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    # 宏平均和微平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='micro', zero_division=0
    )
    
    # 详细分类报告
    class_report = classification_report(true_labels, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'macro_metrics': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro_metrics': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        },
        'per_class_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'classification_report': class_report
    }

def _save_metrics(metrics: Dict[str, Any], output_dir: str):
    """保存指标到文件"""
    metrics_file = os.path.join(output_dir, 'classification_metrics.txt')
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Classification Metrics Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
        
        f.write("Macro Average Metrics:\n")
        f.write(f"Precision: {metrics['macro_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {metrics['macro_metrics']['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['macro_metrics']['f1']:.4f}\n\n")
        
        f.write("Micro Average Metrics:\n")
        f.write(f"Precision: {metrics['micro_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {metrics['micro_metrics']['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['micro_metrics']['f1']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 50 + "\n")
        for i in range(len(metrics['per_class_metrics']['precision'])):
            f.write(f"Class {i}:\n")
            f.write(f"Precision: {metrics['per_class_metrics']['precision'][i]:.4f}\n")
            f.write(f"Recall: {metrics['per_class_metrics']['recall'][i]:.4f}\n")
            f.write(f"F1 Score: {metrics['per_class_metrics']['f1'][i]:.4f}\n")
            f.write(f"Support: {metrics['per_class_metrics']['support'][i]}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(metrics['classification_report'])
    
    # 保存JSON格式的指标
    json_metrics = {
        'accuracy': float(metrics['accuracy']),
        'macro_metrics': {k: float(v) for k, v in metrics['macro_metrics'].items()},
        'micro_metrics': {k: float(v) for k, v in metrics['micro_metrics'].items()},
        'per_class_metrics': {
            k: [float(x) for x in v] if isinstance(v, np.ndarray) else v 
            for k, v in metrics['per_class_metrics'].items()
        }
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metrics saved to {metrics_file}")

def _plot_confusion_matrix(cm: np.ndarray, output_dir: str, label_map: Dict[int, int]):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    
    # 计算百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # 绘制归一化百分比
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def analyze_misclassifications(test_dataset: NewsDataset, predictions: np.ndarray, 
                             true_labels: np.ndarray, output_dir: str) -> Dict[str, Any]:
    """分析错误分类样本"""
    misclassified = []
    error_stats = {}
    
    # 收集错误样本
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != true:
            misclassified.append({
                'index': i,
                'text': test_dataset[i]['text'],
                'true_label': true,
                'predicted_label': pred,
                'original_label': test_dataset[i]['original_label'],
                'text_length': len(test_dataset[i]['text'])
            })
            
            # 统计错误类型
            error_key = f"{true}->{pred}"
            error_stats[error_key] = error_stats.get(error_key, 0) + 1
    
    # 保存分析结果
    _save_misclassification_analysis(misclassified, error_stats, output_dir)
    
    # 分析文本长度对错误的影响
    _analyze_text_length_impact(misclassified, output_dir)
    
    result = {
        'total_samples': len(predictions),
        'misclassified_count': len(misclassified),
        'error_rate': len(misclassified) / len(predictions),
        'error_stats': error_stats,
        'misclassified_examples': misclassified
    }
    
    logger.info(f"Error analysis complete. Error rate: {result['error_rate']:.4f}")
    return result

def _save_misclassification_analysis(misclassified: List[Dict], error_stats: Dict, output_dir: str):
    """保存错误分类分析"""
    with open(os.path.join(output_dir, 'misclassified_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("Misclassified Examples Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # 总体统计
        f.write("Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Misclassified Samples: {len(misclassified)}\n")
        f.write(f"Error Rate: {len(misclassified)/1000:.4f}\n\n")  # 假设总样本数
        
        # 错误类型统计
        sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
        f.write("Error Type Statistics:\n")
        f.write("-" * 30 + "\n")
        for error_type, count in sorted_errors:
            true_label, pred_label = error_type.split("->")
            percentage = count / len(misclassified) * 100
            f.write(f"True {true_label} -> Pred {pred_label}: {count} samples ({percentage:.1f}%)\n")
        
        # 详细样本
        f.write("\n\nDetailed Examples (Top 10):\n")
        f.write("=" * 50 + "\n")
        for i, example in enumerate(misclassified[:10]):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"True Label: {example['true_label']} (Original: {example['original_label']})\n")
            f.write(f"Predicted Label: {example['predicted_label']}\n")
            f.write(f"Text Length: {example['text_length']} characters\n")
            f.write(f"Text: {example['text'][:200]}...\n")
            f.write("-" * 50 + "\n")

def _analyze_text_length_impact(misclassified: List[Dict], output_dir: str):
    """分析文本长度对错误分类的影响"""
    if not misclassified:
        return
    
    lengths = [item['text_length'] for item in misclassified]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Text Lengths in Misclassified Samples')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.0f}')
    plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.0f}')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'text_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def test_prompt_sensitivity(model, tokenizer, test_dataset: NewsDataset, output_dir: str) -> Dict[str, Any]:
    """测试不同提示模板的敏感性"""
    prompt_templates = [
        "Please classify this automotive news article into the appropriate category. Return only the category ID.\nNews text: {text}\nCategory:",
        "Classify the following automotive news article. Return only the category ID.\nArticle: {text}\nCategory:",
        "Given this automotive news article, determine its category. Return only the category ID.\nText: {text}\nCategory:",
        "What category does this automotive news article belong to? Return only the category ID.\nNews: {text}\nCategory:",
        "Analyze this automotive news and provide the category ID.\nContent: {text}\nCategory:"
    ]
    
    results = {}
    original_texts = [test_dataset[i]['text'] for i in range(len(test_dataset))]
    original_labels = [test_dataset[i]['true_label'] for i in range(len(test_dataset))]
    
    for i, template in enumerate(prompt_templates):
        logger.info(f"Testing prompt template {i+1}/{len(prompt_templates)}")
        
        # 创建新数据集
        new_dataset = NewsDataset(
            original_texts,
            original_labels,
            tokenizer,
            prompt_template=template
        )
        
        # 生成预测
        prompt_output_dir = os.path.join(output_dir, f'prompt_{i+1}')
        result = generate_confusion_matrix(model, tokenizer, new_dataset, prompt_output_dir, batch_size=16)
        
        # 分析错误分类
        logger.info(f"Analyzing misclassifications for prompt {i+1}...")
        misclass_result = analyze_misclassifications(
            new_dataset, result['predictions'], result['true_labels'], prompt_output_dir
        )
        
        results[f'prompt_{i+1}'] = {
            'template': template,
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_metrics']['f1'],
            'micro_f1': result['micro_metrics']['f1'],
            'error_rate': misclass_result['error_rate'],
            'misclassified_count': misclass_result['misclassified_count'],
            'error_stats': misclass_result['error_stats']
        }
    
    # 保存对比结果
    _save_prompt_sensitivity_results(results, output_dir)
    
    return results

def _save_prompt_sensitivity_results(results: Dict[str, Any], output_dir: str):
    """保存提示敏感性分析结果"""
    with open(os.path.join(output_dir, 'prompt_sensitivity.txt'), 'w', encoding='utf-8') as f:
        f.write("Prompt Sensitivity Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # 按准确率排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        f.write("Results (sorted by accuracy):\n")
        f.write("-" * 60 + "\n")
        
        for prompt_id, result in sorted_results:
            f.write(f"\n{prompt_id.upper()}:\n")
            f.write(f"Template: {result['template']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Macro F1: {result['macro_f1']:.4f}\n")
            f.write(f"Micro F1: {result['micro_f1']:.4f}\n")
            f.write(f"Error Rate: {result['error_rate']:.4f}\n")
            f.write(f"Misclassified Samples: {result['misclassified_count']}\n")
            
            # 添加错误类型统计
            if result['error_stats']:
                f.write("Top 3 Error Types:\n")
                sorted_errors = sorted(result['error_stats'].items(), key=lambda x: x[1], reverse=True)
                for j, (error_type, count) in enumerate(sorted_errors[:3]):
                    f.write(f"  {j+1}. {error_type}: {count} cases\n")
            
            f.write("-" * 60 + "\n")
        
        # 分析结果
        best_prompt = sorted_results[0]
        worst_prompt = sorted_results[-1]
        
        f.write(f"\nSUMMARY:\n")
        f.write(f"Best performing prompt: {best_prompt[0]} (Accuracy: {best_prompt[1]['accuracy']:.4f})\n")
        f.write(f"Worst performing prompt: {worst_prompt[0]} (Accuracy: {worst_prompt[1]['accuracy']:.4f})\n")
        f.write(f"Performance gap: {best_prompt[1]['accuracy'] - worst_prompt[1]['accuracy']:.4f}\n")
        
        # 错误率比较
        error_rates = [result['error_rate'] for result in results.values()]
        f.write(f"Best Error Rate: {min(error_rates):.4f}\n")
        f.write(f"Worst Error Rate: {max(error_rates):.4f}\n")
        f.write(f"Error Rate Range: {max(error_rates) - min(error_rates):.4f}\n")



def plot_training_losses():
    """绘制训练损失曲线"""
    rank_dirs = glob.glob("training_result/rank_*")
    
    if not rank_dirs:
        logger.warning("No training result directories found")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # 创建综合图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    all_losses = {}
    
    for rank_dir in rank_dirs:
        rank = os.path.basename(rank_dir).split('_')[1]
        loss_file = os.path.join(rank_dir, "training_losses.txt")
        
        if not os.path.exists(loss_file):
            logger.warning(f"Training losses file not found in {rank_dir}")
            continue
        
        steps, losses = _read_loss_file(loss_file)
        
        if not steps:
            continue
        
        all_losses[rank] = {'steps': steps, 'losses': losses}
        
        # 为每个rank创建单独的图
        _plot_single_rank_loss(steps, losses, rank, rank_dir)
    
    # 创建比较图
    if all_losses:
        _plot_loss_comparison(all_losses, "training_result")
    
    logger.info("Training loss plots completed")

def _read_loss_file(loss_file: str) -> Tuple[List[int], List[float]]:
    """读取损失文件"""
    steps = []
    losses = []
    
    try:
        with open(loss_file, 'r') as f:
            for line in f:
                if line.strip() and ': ' in line:
                    try:
                        step_part, loss = line.strip().split(': ')
                        step = int(step_part.replace('Step ', ''))
                        steps.append(step)
                        losses.append(float(loss))
                    except ValueError as e:
                        logger.warning(f"Error parsing line '{line.strip()}': {e}")
                        continue
    except Exception as e:
        logger.error(f"Error reading loss file {loss_file}: {e}")
    
    return steps, losses

def _plot_single_rank_loss(steps: List[int], losses: List[float], rank: str, output_dir: str):
    """绘制单个rank的损失曲线"""
    plt.figure(figsize=(12, 8))
    
    # 原始损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, color='#2E86AB', linewidth=2, alpha=0.8)
    plt.title(f'Training Loss for Rank {rank}', fontsize=14, pad=15)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 添加移动平均
    if len(losses) > 10:
        window_size = min(len(losses) // 10, 50)
        moving_avg = pd.Series(losses).rolling(window=window_size).mean()
        plt.plot(steps, moving_avg, color='#A23B72', linewidth=3, alpha=0.8, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    # 损失分布直方图
    plt.subplot(2, 1, 2)
    plt.hist(losses, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
    plt.title('Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats = {
        'Min': min(losses),
        'Max': max(losses),
        'Mean': np.mean(losses),
        'Std': np.std(losses),
        'Final': losses[-1] if losses else 0
    }
    
    stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats.items()])
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'training_losses_detailed_rank{rank}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed loss plot saved to {output_path}")

def _plot_loss_comparison(all_losses: Dict[str, Dict], output_dir: str):
    """绘制所有rank的损失对比图"""
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_losses)))
    
    for i, (rank, data) in enumerate(all_losses.items()):
        steps = data['steps']
        losses = data['losses']
        
        plt.plot(steps, losses, color=colors[i], linewidth=2, label=f'Rank {rank}', alpha=0.8)
        
        # 添加最终损失标注
        if steps and losses:
            plt.annotate(f'{losses[-1]:.3f}', 
                        xy=(steps[-1], losses[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.title('Training Loss Comparison Across All Ranks', fontsize=16, pad=20)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'training_losses_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss comparison plot saved to {comparison_path}")

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """加载和验证数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # 验证必要列
        required_columns = ['text', 'label_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 数据清理
        df = df.dropna(subset=required_columns)
        df['label_id'] = pd.to_numeric(df['label_id'], errors='coerce')
        df = df.dropna(subset=['label_id'])
        df['label_id'] = df['label_id'].astype(int)
        
        logger.info(f"Data after cleaning: {df.shape}")
        logger.info(f"Label distribution:\n{df['label_id'].value_counts().sort_index()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def main():
    """主函数"""
    try:
        # 创建输出目录
        output_dir = "error_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("Loading test data...")
        df = load_and_validate_data("dataset/news.csv")
        
        test_texts = df['text'].tolist()
        test_labels = df['label_id'].tolist()
        
        # 加载模型
        model_path = "experiment_results/rank_8/checkpoint-4677"
        logger.info(f"Loading model from {model_path}...")
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 创建测试数据集
        logger.info("Creating test dataset...")
        test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
        
        # 生成混淆矩阵
        logger.info("Generating confusion matrix...")
        results = generate_confusion_matrix(model, tokenizer, test_dataset, output_dir)
        
        # 分析错误分类
        logger.info("Analyzing misclassifications...")
        misclass_results = analyze_misclassifications(
            test_dataset, results['predictions'], results['true_labels'], output_dir
        )
        
        # 测试提示敏感性
        logger.info("Testing prompt sensitivity...")
        prompt_results = test_prompt_sensitivity(model, tokenizer, test_dataset, output_dir)
        
        # 绘制训练损失
        logger.info("Plotting training losses...")
        plot_training_losses()
        
        # 保存总结报告
        _save_summary_report(results, misclass_results, prompt_results, output_dir)
        
        logger.info(f"Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

def _save_summary_report(results: Dict, misclass_results: Dict, 
                        prompt_results: Dict, output_dir: str):
    """保存总结报告"""
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("MODEL EVALUATION SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体性能
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Macro F1: {results['macro_metrics']['f1']:.4f}\n")
        f.write(f"Micro F1: {results['micro_metrics']['f1']:.4f}\n")
        f.write(f"Total Samples: {len(results['predictions'])}\n\n")
        
        # 错误分析
        f.write("ERROR ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Misclassified Samples: {misclass_results['misclassified_count']}\n")
        f.write(f"Error Rate: {misclass_results['error_rate']:.4f}\n")
        
        if misclass_results['error_stats']:
            most_common_error = max(misclass_results['error_stats'].items(), key=lambda x: x[1])
            f.write(f"Most Common Error: {most_common_error[0]} ({most_common_error[1]} cases)\n\n")
        
        # 提示敏感性
        if prompt_results:
            f.write("PROMPT SENSITIVITY:\n")
            f.write("-" * 30 + "\n")
            accuracies = [result['accuracy'] for result in prompt_results.values()]
            error_rates = [result['error_rate'] for result in prompt_results.values()]
            
            f.write(f"Best Prompt Accuracy: {max(accuracies):.4f}\n")
            f.write(f"Worst Prompt Accuracy: {min(accuracies):.4f}\n")
            f.write(f"Accuracy Range: {max(accuracies) - min(accuracies):.4f}\n")
            f.write(f"Best Error Rate: {min(error_rates):.4f}\n")
            f.write(f"Worst Error Rate: {max(error_rates):.4f}\n")
            f.write(f"Error Rate Range: {max(error_rates) - min(error_rates):.4f}\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("-" * 30 + "\n")
        f.write("- confusion_matrix.png\n")
        f.write("- classification_metrics.txt\n")
        f.write("- metrics.json\n")
        f.write("- misclassified_analysis.txt\n")
        f.write("- prompt_sensitivity.txt\n")
        f.write("- training loss plots\n")
        f.write("- prompt_*/: Individual analysis for each prompt template\n")

        
if __name__ == "__main__":
    main()