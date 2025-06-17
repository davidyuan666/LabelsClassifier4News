"""
消融研究：分离embedding影响与分类器架构影响
针对Reviewer Comment: "Provide ablation study results isolating embedding impact vs. classifier architecture impact."
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional, Any
import itertools

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
def setup_logging():
    """设置日志配置"""
    log_dir = "ablation_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"bert_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ===================== 数据集定义 =====================
class NewsDatasetBERT(Dataset):
    """新闻数据集类，支持BERT tokenization"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 处理标签映射
        self._process_labels()
        
    def _process_labels(self):
        """处理标签映射"""
        unique_labels = sorted(set(self.labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        self.mapped_labels = [self.label_map[label] for label in self.labels]
        self.num_classes = len(unique_labels)
        
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Label distribution: {dict(pd.Series(self.labels).value_counts().sort_index())}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.mapped_labels[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }

# ===================== Embedding策略定义 =====================
class EmbeddingStrategy(nn.Module):  # 继承nn.Module
    """Embedding策略基类"""
    def __init__(self, name: str):
        super().__init__()  # 调用父类初始化
        self.name = name
    
    def get_embeddings(self, input_ids, attention_mask):
        raise NotImplementedError

class BERTEmbedding(EmbeddingStrategy):
    """BERT预训练embedding"""
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__(f"BERT-{model_name}")
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = self.bert.config.hidden_size
        
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的表示
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return embeddings

class DomainSpecificEmbedding(EmbeddingStrategy):
    """多语言BERT embedding（使用多语言预训练模型）"""
    def __init__(self, model_name: str = 'bert-base-multilingual-cased'):
        super().__init__("Multilingual-BERT")
        # 使用多语言BERT模型，这样可以使用同一个tokenizer
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = self.bert.config.hidden_size
        
        logger.info(f"Using multilingual BERT embeddings: {model_name}")
        
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    

class RandomEmbedding(EmbeddingStrategy):
    """随机初始化embedding（baseline）"""
    def __init__(self, vocab_size: int = 30522, embedding_dim: int = 768):
        super().__init__("Random-Embedding")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def get_embeddings(self, input_ids, attention_mask):
        # 简单的词嵌入平均
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        # 使用attention mask进行加权平均
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class Word2VecEmbedding(EmbeddingStrategy):
    """Word2Vec风格的embedding"""
    def __init__(self, vocab_size: int = 30522, embedding_dim: int = 300):
        super().__init__("Word2Vec-Style")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # 初始化为较小的值，模拟Word2Vec
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
    def get_embeddings(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# ===================== 分类器架构定义 =====================
class ClassifierArchitecture(nn.Module):
    """分类器架构基类"""
    def __init__(self, name: str, input_dim: int, num_classes: int):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.num_classes = num_classes

class TextCNNClassifier(ClassifierArchitecture):
    """TextCNN分类器"""
    def __init__(self, input_dim: int, num_classes: int, num_filters: int = 128, 
                 filter_sizes: List[int] = [3, 4, 5], dropout: float = 0.5):
        super().__init__("TextCNN", input_dim, num_classes)
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # 由于我们接收的是句子级别的embedding，需要重新构造序列维度
        self.projection = nn.Linear(input_dim, input_dim)
        
        # CNN层
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, embeddings):
        # embeddings: [batch_size, embedding_dim]
        # 为了使用CNN，我们需要创建序列维度
        x = embeddings.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        x = x.expand(-1, 10, -1)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # [batch_size, num_filters, new_seq_len]
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LSTMClassifier(ClassifierArchitecture):
    """LSTM分类器"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__("LSTM", input_dim, num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 由于输入是句子级embedding，我们使用简单的方法
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, embeddings):
        # embeddings: [batch_size, embedding_dim]
        x = self.projection(embeddings)  # [batch_size, hidden_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # LSTM处理
        lstm_out, (hidden, _) = self.lstm(x)
        # 使用最后一个时间步的输出
        x = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MLPClassifier(ClassifierArchitecture):
    """多层感知机分类器"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [512, 256], 
                 dropout: float = 0.5):
        super().__init__("MLP", input_dim, num_classes)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, embeddings):
        return self.layers(embeddings)

class LinearClassifier(ClassifierArchitecture):
    """简单线性分类器（baseline）"""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__("Linear", input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, embeddings):
        x = self.dropout(embeddings)
        return self.linear(x)

# ===================== 完整模型定义 =====================
class NewsClassificationModel(nn.Module):
    """新闻分类完整模型"""
    def __init__(self, embedding_strategy: EmbeddingStrategy, 
                 classifier: ClassifierArchitecture):
        super().__init__()
        self.embedding_strategy = embedding_strategy
        self.classifier = classifier
        self.model_name = f"{embedding_strategy.name}+{classifier.name}"
        
    def forward(self, input_ids, attention_mask):
        # 获取embeddings
        embeddings = self.embedding_strategy.get_embeddings(input_ids, attention_mask)
        
        # 分类
        logits = self.classifier(embeddings)
        return logits

# ===================== 训练器 =====================
class AblationTrainer:
    """消融研究训练器"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def train_model(self, model: NewsClassificationModel, train_loader: DataLoader, 
                   val_loader: DataLoader, num_epochs: int = 10, lr: float = 2e-5) -> Dict:
        """训练单个模型"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        train_losses = []
        val_accuracies = []
        
        logger.info(f"Training model: {model.model_name}")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 验证阶段
            val_accuracy = self._evaluate_model(model, val_loader)
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': max(val_accuracies),
            'final_model': model
        }
    
    def _evaluate_model(self, model: NewsClassificationModel, dataloader: DataLoader) -> float:
        """评估模型"""
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    
    def detailed_evaluation(self, model: NewsClassificationModel, 
                          test_loader: DataLoader) -> Dict[str, Any]:
        """详细评估"""
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        # 每个类别的指标
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(true_labels, predictions, average=None, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'per_class_support': per_class_support.tolist(),
            'predictions': predictions,
            'true_labels': true_labels
        }

# ===================== 消融研究主类 =====================
class AblationStudy:
    """消融研究主类"""
    
    def __init__(self, data_path: str, output_dir: str = "ablation_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # 加载数据
        self._load_data()
        
        # 初始化组件
        self._initialize_components()
        
    def _load_data(self):
        """加载和预处理数据"""
        logger.info("Loading and preprocessing data...")
        
        df = pd.read_csv(self.data_path)
        
        # 数据清理
        df = df.dropna(subset=['text', 'label_id'])
        df['label_id'] = pd.to_numeric(df['label_id'], errors='coerce')
        df = df.dropna(subset=['label_id'])
        df['label_id'] = df['label_id'].astype(int)
        
        # 检查类别分布
        label_counts = df['label_id'].value_counts()
        logger.info(f"Original label distribution: {dict(label_counts.sort_index())}")
        
        # 过滤掉样本数量太少的类别（少于2个样本的类别无法进行分层抽样）
        min_samples_per_class = 2
        valid_labels = label_counts[label_counts >= min_samples_per_class].index
        
        if len(valid_labels) < len(label_counts):
            removed_labels = set(label_counts.index) - set(valid_labels)
            logger.warning(f"Removing {len(removed_labels)} classes with less than {min_samples_per_class} samples: {removed_labels}")
            
        # 过滤数据
        df = df[df['label_id'].isin(valid_labels)]
        
        if len(df) == 0:
            raise ValueError("No samples remaining after filtering classes with insufficient samples")
        
        # 重新检查分布
        final_label_counts = df['label_id'].value_counts()
        logger.info(f"Final label distribution: {dict(final_label_counts.sort_index())}")
        logger.info(f"Total samples after filtering: {len(df)}")
        
        # 确保还有足够的数据进行分割
        if len(df) < 10:
            raise ValueError("Insufficient data remaining after filtering")
        
        # 重新映射标签为连续的整数（0, 1, 2, ...）
        unique_labels = sorted(df['label_id'].unique())
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        df['label_id'] = df['label_id'].map(label_mapping)
        
        logger.info(f"Label remapping: {label_mapping}")
        
        try:
            # 分割数据 - 首先分出训练集和临时集
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df['text'].tolist(), df['label_id'].tolist(), 
                test_size=0.4, random_state=42, stratify=df['label_id']
            )
            
            # 再将临时集分为验证集和测试集
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
            )
            
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}")
            logger.info("Falling back to random split without stratification")
            
            # 如果分层抽样失败，使用随机抽样
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df['text'].tolist(), df['label_id'].tolist(), 
                test_size=0.4, random_state=42
            )
            
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=42
            )
        
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        logger.info(f"Data split - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # 验证每个分割中的类别分布
        train_label_dist = pd.Series(train_labels).value_counts().sort_index()
        val_label_dist = pd.Series(val_labels).value_counts().sort_index()
        test_label_dist = pd.Series(test_labels).value_counts().sort_index()
        
        logger.info(f"Train label distribution: {dict(train_label_dist)}")
        logger.info(f"Validation label distribution: {dict(val_label_dist)}")
        logger.info(f"Test label distribution: {dict(test_label_dist)}")


    def _initialize_components(self):
        """初始化embedding策略和分类器架构"""
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 创建数据集（用于获取类别数）
        temp_dataset = NewsDatasetBERT(self.train_texts, self.train_labels, self.tokenizer)
        self.num_classes = temp_dataset.num_classes
        
        # Embedding策略
        self.embedding_strategies = [
            BERTEmbedding('bert-base-uncased'),
            DomainSpecificEmbedding('bert-base-multilingual-cased'),
            RandomEmbedding(),
            Word2VecEmbedding()
        ]
        
        # 分类器架构
        self.classifier_architectures = [
            lambda input_dim: TextCNNClassifier(input_dim, self.num_classes),
            lambda input_dim: LSTMClassifier(input_dim, self.num_classes),
            lambda input_dim: MLPClassifier(input_dim, self.num_classes),
            lambda input_dim: LinearClassifier(input_dim, self.num_classes)
        ]
        
        logger.info(f"Initialized {len(self.embedding_strategies)} embedding strategies")
        logger.info(f"Initialized {len(self.classifier_architectures)} classifier architectures")
        
    def run_ablation_study(self, num_epochs: int = 5, batch_size: int = 32):
        """运行完整的消融研究"""
        logger.info("Starting ablation study...")
        
        # 创建数据加载器
        train_dataset = NewsDatasetBERT(self.train_texts, self.train_labels, self.tokenizer)
        val_dataset = NewsDatasetBERT(self.val_texts, self.val_labels, self.tokenizer)
        test_dataset = NewsDatasetBERT(self.test_texts, self.test_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 训练器
        trainer = AblationTrainer(self.device)
        
        results = {}
        
        # 遍历所有组合
        total_combinations = len(self.embedding_strategies) * len(self.classifier_architectures)
        current_combination = 0
        
        for embedding_strategy in self.embedding_strategies:
            for classifier_factory in self.classifier_architectures:
                current_combination += 1
                logger.info(f"Training combination {current_combination}/{total_combinations}")
                
                # 创建分类器
                classifier = classifier_factory(embedding_strategy.embedding_dim)
                
                # 创建完整模型
                model = NewsClassificationModel(embedding_strategy, classifier)
                
                # 训练模型
                try:
                    train_result = trainer.train_model(model, train_loader, val_loader, num_epochs)
                    
                    # 在测试集上评估
                    test_result = trainer.detailed_evaluation(train_result['final_model'], test_loader)
                    
                    # 保存结果
                    model_key = model.model_name
                    results[model_key] = {
                        'embedding_strategy': embedding_strategy.name,
                        'classifier_architecture': classifier.name,
                        'train_result': {
                            'best_val_accuracy': train_result['best_val_accuracy'],
                            'final_train_loss': train_result['train_losses'][-1],
                        },
                        'test_result': test_result
                    }
                    
                    logger.info(f"Completed {model_key}: Test Accuracy = {test_result['accuracy']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model.model_name}: {e}")
                    continue
        
        self.results = results
        
        # 保存结果
        self._save_results()
        
        # 生成分析报告
        self._generate_analysis_report()
        
        # 生成可视化
        self._generate_visualizations()
        
        logger.info("Ablation study completed!")
        
    def _save_results(self):
        """保存结果到JSON文件"""
        results_file = os.path.join(self.output_dir, 'ablation_results.json')
        
        # 准备可序列化的结果
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = {
                'embedding_strategy': value['embedding_strategy'],
                'classifier_architecture': value['classifier_architecture'],
                'train_result': value['train_result'],
                'test_result': {
                    k: v for k, v in value['test_result'].items() 
                    if k not in ['predictions', 'true_labels']  # 排除大数组
                }
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_file}")
        
    def _generate_analysis_report(self):
        """生成分析报告"""
        report_file = os.path.join(self.output_dir, 'ablation_analysis_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ABLATION STUDY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Check if we have any results
            if not self.results:
                f.write("ERROR: No successful training results found!\n")
                f.write("All model training attempts failed. Please check the training logs for details.\n")
                logger.warning("No results to analyze - all training attempts failed")
                return
            
            # 按不同指标排序
            metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
            
            f.write("1. OVERALL RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for metric in metrics:
                sorted_results = sorted(self.results.items(), 
                                    key=lambda x: x[1]['test_result'][metric], reverse=True)
                
                f.write(f"\nTop 5 performing combinations by {metric.upper()}:\n")
                for i, (model_name, result) in enumerate(sorted_results[:5]):
                    f.write(f"{i+1}. {model_name}: {result['test_result'][metric]:.4f}\n")
            f.write("\n")
            
            # 2. Embedding策略分析 - 多指标
            f.write("2. EMBEDDING STRATEGY ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for metric in metrics:
                embedding_performance = {}
                for model_name, result in self.results.items():
                    embedding = result['embedding_strategy']
                    if embedding not in embedding_performance:
                        embedding_performance[embedding] = []
                    embedding_performance[embedding].append(result['test_result'][metric])
                
                f.write(f"\nAverage {metric} by embedding strategy:\n")
                sorted_embeddings = sorted(embedding_performance.items(), 
                                         key=lambda x: np.mean(x[1]), reverse=True)
                for embedding, values in sorted_embeddings:
                    avg_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"- {embedding}: {avg_val:.4f} (±{std_val:.4f})\n")
            f.write("\n")
            
            # 3. 分类器架构分析 - 多指标
            f.write("3. CLASSIFIER ARCHITECTURE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for metric in metrics:
                classifier_performance = {}
                for model_name, result in self.results.items():
                    classifier = result['classifier_architecture']
                    if classifier not in classifier_performance:
                        classifier_performance[classifier] = []
                    classifier_performance[classifier].append(result['test_result'][metric])
                
                f.write(f"\nAverage {metric} by classifier architecture:\n")
                sorted_classifiers = sorted(classifier_performance.items(), 
                                          key=lambda x: np.mean(x[1]), reverse=True)
                for classifier, values in sorted_classifiers:
                    avg_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"- {classifier}: {avg_val:.4f} (±{std_val:.4f})\n")
            f.write("\n")
            
            # 4. 最佳组合分析 - 多指标
            f.write("4. BEST COMBINATION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            # 计算综合排名
            composite_scores = {}
            for model_name, result in self.results.items():
                # 使用加权平均计算综合分数
                composite_score = (
                    result['test_result']['accuracy'] * 0.3 +
                    result['test_result']['macro_f1'] * 0.4 +
                    result['test_result']['macro_precision'] * 0.15 +
                    result['test_result']['macro_recall'] * 0.15
                )
                composite_scores[model_name] = composite_score
            
            best_model = max(composite_scores.items(), key=lambda x: x[1])
            best_result = self.results[best_model[0]]
            
            f.write(f"Best combination (composite score): {best_model[0]}\n")
            f.write(f"- Composite Score: {best_model[1]:.4f}\n")
            f.write(f"- Accuracy: {best_result['test_result']['accuracy']:.4f}\n")
            f.write(f"- Macro F1: {best_result['test_result']['macro_f1']:.4f}\n")
            f.write(f"- Macro Precision: {best_result['test_result']['macro_precision']:.4f}\n")
            f.write(f"- Macro Recall: {best_result['test_result']['macro_recall']:.4f}\n")
            f.write(f"- Embedding: {best_result['embedding_strategy']}\n")
            f.write(f"- Classifier: {best_result['classifier_architecture']}\n\n")
            
            # 5. 详细性能表格
            f.write("5. DETAILED PERFORMANCE TABLE\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<40} {'Acc':<6} {'F1':<6} {'Pre':<6} {'Rec':<6}\n")
            f.write("-" * 68 + "\n")
            
            sorted_by_f1 = sorted(self.results.items(), 
                                key=lambda x: x[1]['test_result']['macro_f1'], reverse=True)
            
            for model_name, result in sorted_by_f1:
                acc = result['test_result']['accuracy']
                f1 = result['test_result']['macro_f1']
                pre = result['test_result']['macro_precision']
                rec = result['test_result']['macro_recall']
                f.write(f"{model_name:<40} {acc:<6.3f} {f1:<6.3f} {pre:<6.3f} {rec:<6.3f}\n")
            f.write("\n")
            
            # 6. 统计显著性分析 - 多指标
            f.write("6. STATISTICAL ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for metric in metrics:
                # Embedding策略的方差分析
                embedding_performance = {}
                for result in self.results.values():
                    embedding = result['embedding_strategy']
                    if embedding not in embedding_performance:
                        embedding_performance[embedding] = []
                    embedding_performance[embedding].append(result['test_result'][metric])
                
                embedding_values = list(embedding_performance.values())
                if len(embedding_values) > 1:
                    from scipy import stats
                    f_stat, p_value = stats.f_oneway(*embedding_values)
                    f.write(f"Embedding strategies ANOVA ({metric}): F={f_stat:.4f}, p={p_value:.4f}\n")
                
                # 分类器架构的方差分析
                classifier_performance = {}
                for result in self.results.values():
                    classifier = result['classifier_architecture']
                    if classifier not in classifier_performance:
                        classifier_performance[classifier] = []
                    classifier_performance[classifier].append(result['test_result'][metric])
                
                classifier_values = list(classifier_performance.values())
                if len(classifier_values) > 1:
                    from scipy import stats
                    f_stat, p_value = stats.f_oneway(*classifier_values)
                    f.write(f"Classifier architectures ANOVA ({metric}): F={f_stat:.4f}, p={p_value:.4f}\n")
                
                f.write("\n")
        
        logger.info(f"Analysis report saved to {report_file}")
        
    def _generate_visualizations(self):
        """生成可视化图表"""
        # Check if we have any results
        if not self.results:
            logger.warning("No results to visualize - skipping visualization generation")
            return
        
        # 1. 多指标性能热力图
        self._plot_multi_metric_heatmaps()
        
        # 2. 组件贡献分析 - 多指标
        self._plot_multi_metric_component_contribution()
        
        # 3. 详细性能对比 - 多指标
        self._plot_multi_metric_detailed_comparison()
        
        # 4. 雷达图对比
        self._plot_radar_comparison()
        
    def _plot_multi_metric_heatmaps(self):
        """绘制多指标性能热力图"""
        if not self.results:
            return
            
        metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        # 准备数据
        embeddings = sorted(set(r['embedding_strategy'] for r in self.results.values()))
        classifiers = sorted(set(r['classifier_architecture'] for r in self.results.values()))
        
        if not embeddings or not classifiers:
            logger.warning("Insufficient data for heatmap visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            heatmap_data = np.zeros((len(embeddings), len(classifiers)))
            
            for i, embedding in enumerate(embeddings):
                for j, classifier in enumerate(classifiers):
                    for result in self.results.values():
                        if (result['embedding_strategy'] == embedding and 
                            result['classifier_architecture'] == classifier):
                            heatmap_data[i, j] = result['test_result'][metric]
                            break
            
            sns.heatmap(heatmap_data, 
                    xticklabels=classifiers, 
                    yticklabels=embeddings,
                    annot=True, 
                    fmt='.3f', 
                    cmap='YlOrRd',
                    cbar_kws={'label': label},
                    ax=axes[idx])
            axes[idx].set_title(f'Ablation Study: {label} Heatmap')
            axes[idx].set_xlabel('Classifier Architecture')
            axes[idx].set_ylabel('Embedding Strategy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multi_metric_heatmaps.png'), dpi=300)
        plt.close()

    def _plot_multi_metric_component_contribution(self):
        """绘制多指标组件贡献分析"""
        if not self.results:
            return
            
        metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Embedding策略贡献
            embedding_performance = {}
            for result in self.results.values():
                embedding = result['embedding_strategy']
                if embedding not in embedding_performance:
                    embedding_performance[embedding] = []
                embedding_performance[embedding].append(result['test_result'][metric])
            
            if not embedding_performance:
                continue
            
            embeddings = list(embedding_performance.keys())
            embedding_means = [np.mean(embedding_performance[e]) for e in embeddings]
            embedding_stds = [np.std(embedding_performance[e]) for e in embeddings]
            
            axes[0, col].bar(embeddings, embedding_means, yerr=embedding_stds, capsize=5)
            axes[0, col].set_title(f'Embedding Strategy {label}')
            axes[0, col].set_ylabel(label)
            axes[0, col].tick_params(axis='x', rotation=45)
            
            # 分类器架构贡献
            classifier_performance = {}
            for result in self.results.values():
                classifier = result['classifier_architecture']
                if classifier not in classifier_performance:
                    classifier_performance[classifier] = []
                classifier_performance[classifier].append(result['test_result'][metric])
            
            classifiers = list(classifier_performance.keys())
            classifier_means = [np.mean(classifier_performance[c]) for c in classifiers]
            classifier_stds = [np.std(classifier_performance[c]) for c in classifiers]
            
            axes[1, col].bar(classifiers, classifier_means, yerr=classifier_stds, capsize=5)
            axes[1, col].set_title(f'Classifier Architecture {label}')
            axes[1, col].set_ylabel(label)
            axes[1, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multi_metric_component_contribution.png'), dpi=300)
        plt.close()

    def _plot_multi_metric_detailed_comparison(self):
        """绘制多指标详细性能对比"""
        if not self.results:
            return
            
        metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        
        # 准备数据
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, result in self.results.items():
            model_names.append(model_name.replace('+', '\n+\n'))
            for metric in metrics:
                metric_values[metric].append(result['test_result'][metric])
        
        if not model_names:
            logger.warning("No data for detailed comparison visualization")
            return
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            bars = axes[idx].bar(range(len(model_names)), metric_values[metric])
            axes[idx].set_title(f'{label} Comparison')
            axes[idx].set_ylabel(label)
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'multi_metric_detailed_comparison.png'), dpi=300)
        plt.close()

    def _plot_radar_comparison(self):
        """绘制雷达图对比不同模型的多指标性能"""
        if not self.results:
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        
        metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']
        
        # 选择top 5模型
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_result']['macro_f1'], reverse=True)[:5]
        
        if len(sorted_models) == 0:
            return
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 完成圆形
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, (model_name, result) in enumerate(sorted_models):
            values = [result['test_result'][metric] for metric in metrics]
            values += values[:1]  # 完成圆形
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name.replace('+', '+\n'), color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.title('Top 5 Models: Multi-Metric Performance Radar Chart', size=16, y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ===================== 主函数 =====================
def main():
    """主函数"""
    try:
        # 初始化消融研究
        study = AblationStudy(
            data_path="dataset/news.csv",
            output_dir="ablation_results"
        )
        
        # 运行消融研究
        study.run_ablation_study(
            num_epochs=10,  # 可以根据需要调整
            batch_size=32
        )
        
        logger.info("Ablation study completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 