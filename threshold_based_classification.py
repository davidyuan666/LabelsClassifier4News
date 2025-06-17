"""
基于阈值的分层分类策略
根据样本数量设置阈值，剔除高频类别进行分类对比
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
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
from collections import Counter

# 忽略警告
warnings.filterwarnings('ignore')

# 设置日志
def setup_logging():
    """设置日志配置"""
    log_dir = "threshold_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

# ===================== 基于阈值的数据分析器 =====================
class ThresholdBasedAnalyzer:
    """基于阈值的数据分析器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        logger.info("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # 数据清理
        self.df = self.df.dropna(subset=['text', 'label_id'])
        self.df['label_id'] = pd.to_numeric(self.df['label_id'], errors='coerce')
        self.df = self.df.dropna(subset=['label_id'])
        self.df['label_id'] = self.df['label_id'].astype(int)
        
        logger.info(f"Total samples: {len(self.df)}")
        
    def analyze_label_distribution(self):
        """分析标签分布并建议阈值"""
        logger.info("Analyzing label distribution...")
        
        label_counts = self.df['label_id'].value_counts().sort_index()
        total_samples = len(self.df)
        
        print("\n" + "="*70)
        print("THRESHOLD-BASED CLASSIFICATION ANALYSIS")
        print("="*70)
        
        # 详细标签分布
        print(f"\nTotal samples: {total_samples}")
        print(f"Number of unique labels: {len(label_counts)}")
        print("\nLabel distribution (sorted by sample count):")
        
        sorted_by_count = label_counts.sort_values(ascending=False)
        for i, (label, count) in enumerate(sorted_by_count.items()):
            percentage = count / total_samples * 100
            print(f"  {i+1:2d}. Label {label}: {count:,} samples ({percentage:.2f}%)")
        
        # 分析推荐的阈值
        self._suggest_thresholds(sorted_by_count, total_samples)
        
        return label_counts
    
    def _suggest_thresholds(self, sorted_counts, total_samples):
        """建议阈值设置"""
        print(f"\n" + "="*50)
        print("SUGGESTED THRESHOLD STRATEGIES")
        print("="*50)
        
        # 计算统计信息
        mean_count = sorted_counts.mean()
        median_count = sorted_counts.median()
        std_count = sorted_counts.std()
        
        print(f"\nStatistical summary:")
        print(f"  Mean samples per label: {mean_count:.1f}")
        print(f"  Median samples per label: {median_count:.1f}")
        print(f"  Standard deviation: {std_count:.1f}")
        
        # 策略1: 基于百分位数的阈值
        print(f"\nStrategy 1 - Percentile-based thresholds:")
        percentiles = [75, 85, 90, 95]
        for p in percentiles:
            threshold = sorted_counts.quantile(1 - p/100)
            excluded_labels = sorted_counts[sorted_counts > threshold].index.tolist()
            excluded_samples = sorted_counts[sorted_counts > threshold].sum()
            remaining_samples = total_samples - excluded_samples
            remaining_labels = len(sorted_counts) - len(excluded_labels)
            
            print(f"  {p}th percentile (threshold > {threshold:.0f}):")
            print(f"    Exclude {len(excluded_labels)} labels ({excluded_samples:,} samples, {excluded_samples/total_samples*100:.1f}%)")
            print(f"    Remaining: {remaining_labels} labels ({remaining_samples:,} samples, {remaining_samples/total_samples*100:.1f}%)")
        
        # 策略2: 基于标准差的阈值
        print(f"\nStrategy 2 - Standard deviation-based thresholds:")
        for multiplier in [1, 1.5, 2, 2.5]:
            threshold = mean_count + multiplier * std_count
            excluded_labels = sorted_counts[sorted_counts > threshold].index.tolist()
            excluded_samples = sorted_counts[sorted_counts > threshold].sum()
            remaining_samples = total_samples - excluded_samples
            remaining_labels = len(sorted_counts) - len(excluded_labels)
            
            print(f"  Mean + {multiplier}*std (threshold > {threshold:.0f}):")
            print(f"    Exclude {len(excluded_labels)} labels ({excluded_samples:,} samples, {excluded_samples/total_samples*100:.1f}%)")
            print(f"    Remaining: {remaining_labels} labels ({remaining_samples:,} samples, {remaining_samples/total_samples*100:.1f}%)")
        
        # 策略3: 固定样本数阈值
        print(f"\nStrategy 3 - Fixed sample count thresholds:")
        max_count = sorted_counts.max()
        fixed_thresholds = []
        
        # 动态生成固定阈值
        if max_count > 10000:
            fixed_thresholds = [5000, 10000, 15000, 20000]
        elif max_count > 5000:
            fixed_thresholds = [2000, 3000, 4000, 5000]
        elif max_count > 1000:
            fixed_thresholds = [500, 800, 1000, 1500]
        else:
            fixed_thresholds = [100, 200, 300, 500]
        
        for threshold in fixed_thresholds:
            if threshold < max_count:
                excluded_labels = sorted_counts[sorted_counts > threshold].index.tolist()
                excluded_samples = sorted_counts[sorted_counts > threshold].sum()
                remaining_samples = total_samples - excluded_samples
                remaining_labels = len(sorted_counts) - len(excluded_labels)
                
                print(f"  Fixed threshold > {threshold}:")
                print(f"    Exclude {len(excluded_labels)} labels ({excluded_samples:,} samples, {excluded_samples/total_samples*100:.1f}%)")
                print(f"    Remaining: {remaining_labels} labels ({remaining_samples:,} samples, {remaining_samples/total_samples*100:.1f}%)")
        
        return {
            'percentile_thresholds': percentiles,
            'std_multipliers': [1, 1.5, 2, 2.5],
            'fixed_thresholds': fixed_thresholds,
            'statistics': {
                'mean': mean_count,
                'median': median_count,
                'std': std_count,
                'max': max_count
            }
        }

# ===================== 简化的分类器 =====================
class SimpleBERTClassifier(nn.Module):
    """简化的BERT分类器"""
    
    def __init__(self, num_classes: int, model_name: str = 'bert-base-uncased', dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

# ===================== 简化的数据集 =====================
class SimpleNewsDataset(Dataset):
    """简化的新闻数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ===================== 阈值实验类 =====================
class ThresholdExperiment:
    """基于阈值的分类实验"""
    
    def __init__(self, data_path: str, output_dir: str = "threshold_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # 分析数据
        self.analyzer = ThresholdBasedAnalyzer(data_path)
        
    def run_threshold_comparison(self, num_epochs: int = 5, batch_size: int = 32):
        """运行阈值对比实验"""
        logger.info("Starting threshold-based classification experiment...")
        
        # 1. 分析数据分布
        label_counts = self.analyzer.analyze_label_distribution()
        sorted_counts = label_counts.sort_values(ascending=False)
        
        # 2. 准备数据
        texts = self.analyzer.df['text'].tolist()
        labels = self.analyzer.df['label_id'].tolist()
        
        # 数据分割
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.4, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )
        
        # 3. 定义阈值策略
        thresholds = self._generate_threshold_strategies(sorted_counts)
        
        # 4. 运行实验
        results = {}
        
        # 参考实验：使用全部数据
        print(f"\n" + "="*60)
        print("RUNNING REFERENCE EXPERIMENT (ALL DATA)")
        print("="*60)
        
        results['reference_all_data'] = self._run_classification_with_threshold(
            train_texts, train_labels, val_texts, val_labels, 
            test_texts, test_labels, threshold=None, 
            name="All Data", num_epochs=num_epochs, batch_size=batch_size
        )
        
        # 阈值实验
        for i, (threshold_value, threshold_name) in enumerate(thresholds):
            print(f"\n" + "="*60)
            print(f"RUNNING THRESHOLD EXPERIMENT {i+1}: {threshold_name}")
            print("="*60)
            
            results[f'threshold_{i+1}_{threshold_name.lower().replace(" ", "_")}'] = self._run_classification_with_threshold(
                train_texts, train_labels, val_texts, val_labels, 
                test_texts, test_labels, threshold=threshold_value, 
                name=threshold_name, num_epochs=num_epochs, batch_size=batch_size
            )
        
        # 5. 分析和保存结果
        self._analyze_threshold_results(results)
        self._save_threshold_results(results)
        self._plot_threshold_comparison(results)
        
        logger.info("Threshold-based classification experiment completed!")
        
        return results
    
    def _generate_threshold_strategies(self, sorted_counts):
        """生成阈值策略"""
        total_samples = sorted_counts.sum()
        mean_count = sorted_counts.mean()
        std_count = sorted_counts.std()
        
        thresholds = []
        
        # 基于百分位数的阈值
        for p in [90, 95, 99]:
            threshold = sorted_counts.quantile(1 - p/100)
            thresholds.append((threshold, f"P{p} (>{threshold:.0f})"))
        
        # 基于标准差的阈值  
        for multiplier in [1.5, 2, 3]:
            threshold = mean_count + multiplier * std_count
            thresholds.append((threshold, f"Mean+{multiplier}*STD (>{threshold:.0f})"))
        
        # 固定阈值
        max_count = sorted_counts.max()
        if max_count > 5000:
            fixed_thresholds = [2000, 5000, 10000]
        elif max_count > 1000:
            fixed_thresholds = [500, 1000, 2000]
        else:
            fixed_thresholds = [100, 300, 500]
        
        for threshold in fixed_thresholds:
            if threshold < max_count:
                thresholds.append((threshold, f"Fixed >{threshold}"))
        
        return thresholds
    
    def _run_classification_with_threshold(self, train_texts, train_labels, val_texts, val_labels,
                                         test_texts, test_labels, threshold=None, 
                                         name="", num_epochs=5, batch_size=32):
        """运行带阈值的分类实验"""
        
        if threshold is not None:
            # 计算每个标签的样本数
            label_counts = Counter(train_labels + val_labels + test_labels)
            
            # 找出需要剔除的高频标签
            excluded_labels = [label for label, count in label_counts.items() if count > threshold]
            
            # 过滤数据
            train_mask = [label not in excluded_labels for label in train_labels]
            val_mask = [label not in excluded_labels for label in val_labels]
            test_mask = [label not in excluded_labels for label in test_labels]
            
            filtered_train_texts = [text for text, mask in zip(train_texts, train_mask) if mask]
            filtered_train_labels = [label for label, mask in zip(train_labels, train_mask) if mask]
            filtered_val_texts = [text for text, mask in zip(val_texts, val_mask) if mask]
            filtered_val_labels = [label for label, mask in zip(val_labels, val_mask) if mask]
            filtered_test_texts = [text for text, mask in zip(test_texts, test_mask) if mask]
            filtered_test_labels = [label for label, mask in zip(test_labels, test_mask) if mask]
            
            print(f"Excluded {len(excluded_labels)} high-frequency labels: {excluded_labels}")
            print(f"Original samples: {len(train_texts + val_texts + test_texts)}")
            print(f"Remaining samples: {len(filtered_train_texts + filtered_val_texts + filtered_test_texts)}")
            
        else:
            # 使用全部数据
            filtered_train_texts = train_texts
            filtered_train_labels = train_labels
            filtered_val_texts = val_texts
            filtered_val_labels = val_labels
            filtered_test_texts = test_texts
            filtered_test_labels = test_labels
            excluded_labels = []
        
        # 重新映射标签为连续整数
        unique_labels = sorted(set(filtered_train_labels + filtered_val_labels + filtered_test_labels))
        if len(unique_labels) < 2:
            print(f"Warning: Only {len(unique_labels)} classes remaining, skipping...")
            return None
            
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        mapped_train_labels = [label_map[label] for label in filtered_train_labels]
        mapped_val_labels = [label_map[label] for label in filtered_val_labels]
        mapped_test_labels = [label_map[label] for label in filtered_test_labels]
        
        num_classes = len(unique_labels)
        print(f"Training with {num_classes} classes")
        
        # 创建数据集和数据加载器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        train_dataset = SimpleNewsDataset(filtered_train_texts, mapped_train_labels, tokenizer)
        val_dataset = SimpleNewsDataset(filtered_val_texts, mapped_val_labels, tokenizer)
        test_dataset = SimpleNewsDataset(filtered_test_texts, mapped_test_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 训练模型
        model = SimpleBERTClassifier(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        
        # 训练循环
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # 训练
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
            
            # 验证
            val_accuracy = self._evaluate_model(model, val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        # 测试评估
        test_result = self._detailed_evaluation(model, test_loader, num_classes)
        
        return {
            'threshold': threshold,
            'threshold_name': name,
            'excluded_labels': excluded_labels,
            'num_classes': num_classes,
            'num_samples': len(filtered_train_texts + filtered_val_texts + filtered_test_texts),
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_result': test_result
        }
    
    def _evaluate_model(self, model, dataloader):
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        return correct / total
    
    def _detailed_evaluation(self, model, test_loader, num_classes):
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
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1
        }
    
    def _analyze_threshold_results(self, results):
        """分析阈值结果"""
        print(f"\n" + "="*80)
        print("THRESHOLD COMPARISON ANALYSIS")
        print("="*80)
        
        # 准备数据
        analysis_data = []
        for name, result in results.items():
            if result is not None:
                analysis_data.append({
                    'name': result['threshold_name'],
                    'threshold': result['threshold'],
                    'num_classes': result['num_classes'],
                    'num_samples': result['num_samples'],
                    'accuracy': result['test_result']['accuracy'],
                    'macro_f1': result['test_result']['macro_f1'],
                    'macro_precision': result['test_result']['macro_precision'],
                    'macro_recall': result['test_result']['macro_recall']
                })
        
        # 按F1分数排序
        analysis_data.sort(key=lambda x: x['macro_f1'], reverse=True)
        
        print(f"\nResults ranked by Macro F1 Score:")
        print(f"{'Rank':<4} {'Strategy':<20} {'Classes':<8} {'Samples':<10} {'Accuracy':<10} {'Macro F1':<10}")
        print("-" * 70)
        
        for i, data in enumerate(analysis_data):
            print(f"{i+1:<4} {data['name']:<20} {data['num_classes']:<8} {data['num_samples']:<10} "
                  f"{data['accuracy']:<10.4f} {data['macro_f1']:<10.4f}")
        
        # 分析最佳策略
        if analysis_data:
            best_strategy = analysis_data[0]
            reference = next((d for d in analysis_data if d['name'] == 'All Data'), None)
            
            print(f"\n" + "="*50)
            print("KEY FINDINGS")
            print("="*50)
            
            print(f"\nBest performing strategy: {best_strategy['name']}")
            print(f"  - Macro F1: {best_strategy['macro_f1']:.4f}")
            print(f"  - Accuracy: {best_strategy['accuracy']:.4f}")
            print(f"  - Classes: {best_strategy['num_classes']}")
            print(f"  - Samples: {best_strategy['num_samples']:,}")
            
            if reference and best_strategy['name'] != 'All Data':
                f1_improvement = best_strategy['macro_f1'] - reference['macro_f1']
                acc_improvement = best_strategy['accuracy'] - reference['accuracy']
                sample_reduction = reference['num_samples'] - best_strategy['num_samples']
                class_reduction = reference['num_classes'] - best_strategy['num_classes']
                
                print(f"\nImprovement over using all data:")
                print(f"  - Macro F1 improvement: {f1_improvement:+.4f}")
                print(f"  - Accuracy improvement: {acc_improvement:+.4f}")
                print(f"  - Sample reduction: {sample_reduction:,} ({sample_reduction/reference['num_samples']*100:.1f}%)")
                print(f"  - Class reduction: {class_reduction} ({class_reduction/reference['num_classes']*100:.1f}%)")
    
    def _save_threshold_results(self, results):
        """保存结果"""
        results_file = os.path.join(self.output_dir, 'threshold_results.json')
        
        # 准备可序列化的结果
        serializable_results = {}
        for name, result in results.items():
            if result is not None:
                serializable_results[name] = {
                    k: v for k, v in result.items() 
                    if k not in ['train_losses', 'val_accuracies']  # 排除训练历史
                }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_file}")
    
    def _plot_threshold_comparison(self, results):
        """绘制阈值对比图"""
        # 准备绘图数据
        names = []
        accuracies = []
        f1_scores = []
        num_classes = []
        num_samples = []
        
        for name, result in results.items():
            if result is not None:
                names.append(result['threshold_name'])
                accuracies.append(result['test_result']['accuracy'])
                f1_scores.append(result['test_result']['macro_f1'])
                num_classes.append(result['num_classes'])
                num_samples.append(result['num_samples'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 准确率对比
        bars1 = axes[0, 0].bar(range(len(names)), accuracies)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, value in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. F1分数对比
        bars2 = axes[0, 1].bar(range(len(names)), f1_scores)
        axes[0, 1].set_title('Macro F1 Score Comparison')
        axes[0, 1].set_ylabel('Macro F1')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        
        for bar, value in zip(bars2, f1_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 类别数量对比
        axes[1, 0].bar(range(len(names)), num_classes)
        axes[1, 0].set_title('Number of Classes')
        axes[1, 0].set_ylabel('Classes')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        
        # 4. 样本数量对比
        axes[1, 1].bar(range(len(names)), [n/1000 for n in num_samples])  # 转换为千
        axes[1, 1].set_title('Number of Samples (thousands)')
        axes[1, 1].set_ylabel('Samples (K)')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 散点图：样本数 vs 性能
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # F1 vs 样本数
        ax1.scatter([n/1000 for n in num_samples], f1_scores, s=100, alpha=0.7)
        ax1.set_xlabel('Number of Samples (thousands)')
        ax1.set_ylabel('Macro F1 Score')
        ax1.set_title('Performance vs Sample Size')
        ax1.grid(True, alpha=0.3)
        
        # 添加标签
        for i, name in enumerate(names):
            ax1.annotate(name, (num_samples[i]/1000, f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # F1 vs 类别数
        ax2.scatter(num_classes, f1_scores, s=100, alpha=0.7)
        ax2.set_xlabel('Number of Classes')
        ax2.set_ylabel('Macro F1 Score')
        ax2.set_title('Performance vs Number of Classes')
        ax2.grid(True, alpha=0.3)
        
        for i, name in enumerate(names):
            ax2.annotate(name, (num_classes[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_vs_size.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ===================== 主函数 =====================
def main():
    """主函数"""
    try:
        # 创建实验实例
        experiment = ThresholdExperiment(
            data_path="dataset/news.csv",
            output_dir="threshold_results"
        )
        
        # 运行阈值对比实验
        results = experiment.run_threshold_comparison(
            num_epochs=5,  # 可以根据需要调整
            batch_size=32
        )
        
        print(f"\n" + "="*80)
        print("EXPERIMENT COMPLETED!")
        print("="*80)
        print(f"Results saved in: threshold_results/")
        print(f"Check threshold_comparison.png for visual comparison")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 