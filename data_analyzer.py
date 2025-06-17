"""
精确的数据集分析工具
分析新闻数据集的类别和样本数量
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import sys

def analyze_dataset(data_path="dataset/news.csv"):
    """分析数据集"""
    print("🔍 正在分析数据集...")
    print("="*70)
    
    try:
        # 读取CSV文件
        print(f"📂 读取文件: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ 文件读取成功!")
        
        # 显示文件基本信息
        print(f"\n📊 文件基本信息:")
        print(f"   原始行数: {len(df):,}")
        print(f"   列数: {len(df.columns)}")
        print(f"   列名: {list(df.columns)}")
        
        # 检查是否有text和label_id列
        if 'text' not in df.columns or 'label_id' not in df.columns:
            print("\n⚠️  警告: 找不到'text'或'label_id'列")
            print(f"可用列: {list(df.columns)}")
            # 尝试自动识别可能的列
            text_candidates = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
            label_candidates = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'id' in col.lower()]
            
            if text_candidates:
                print(f"可能的文本列: {text_candidates}")
            if label_candidates:
                print(f"可能的标签列: {label_candidates}")
            return
        
        # 数据清理前的统计
        original_size = len(df)
        
        # 清理空值
        print(f"\n🧹 清理数据...")
        df_clean = df.dropna(subset=['text', 'label_id']).copy()
        after_dropna = len(df_clean)
        
        # 转换label_id为数值类型
        df_clean['label_id'] = pd.to_numeric(df_clean['label_id'], errors='coerce')
        df_clean = df_clean.dropna(subset=['label_id'])
        df_clean['label_id'] = df_clean['label_id'].astype(int)
        
        final_size = len(df_clean)
        
        print(f"   清理前: {original_size:,} 行")
        print(f"   清理空值后: {after_dropna:,} 行")
        print(f"   最终可用: {final_size:,} 行")
        
        if final_size < original_size:
            print(f"   ⚠️  丢失了 {original_size - final_size:,} 行数据")
        
        # 统计标签信息
        print(f"\n🏷️  标签统计:")
        unique_labels = df_clean['label_id'].nunique()
        label_counts = df_clean['label_id'].value_counts()
        
        print(f"   标签类别总数: {unique_labels:,}")
        print(f"   样本总数: {final_size:,}")
        print(f"   平均每类样本数: {final_size/unique_labels:.1f}")
        print(f"   标签ID范围: {df_clean['label_id'].min()} - {df_clean['label_id'].max()}")
        
        # 样本分布统计
        print(f"\n📈 样本分布:")
        print(f"   最多样本的类别: {label_counts.max():,} 样本")
        print(f"   最少样本的类别: {label_counts.min():,} 样本")
        print(f"   中位数: {label_counts.median():.0f} 样本")
        print(f"   标准差: {label_counts.std():.1f}")
        print(f"   不平衡比例: {label_counts.max()/label_counts.min():.1f}:1")
        
        # 样本量分布区间分析
        print(f"\n📊 按样本量分组的类别分布:")
        bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, float('inf')]
        labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', '500+']
        
        binned = pd.cut(label_counts, bins=bins, labels=labels, include_lowest=True)
        bin_counts = binned.value_counts().sort_index()
        
        print(f"   {'样本量范围':<12} {'类别数量':<8} {'占比':<8}")
        print("   " + "-" * 30)
        
        for bin_range, count in bin_counts.items():
            if count > 0:
                percentage = count / unique_labels * 100
                print(f"   {bin_range:<12} {count:<8} {percentage:<7.1f}%")
        
        # 显示前20个最多样本的类别
        print(f"\n📋 样本数最多的前20个类别:")
        print(f"   {'排名':<4} {'标签ID':<8} {'样本数':<8} {'占比':<8}")
        print("   " + "-" * 35)
        
        top_20 = label_counts.head(20)
        for i, (label_id, count) in enumerate(top_20.items(), 1):
            percentage = count / final_size * 100
            print(f"   {i:<4} {label_id:<8} {count:<8} {percentage:<7.2f}%")
        
        # 阈值策略建议
        print(f"\n💡 阈值策略分析:")
        thresholds = [1, 2, 5, 10, 20, 50, 100, 200]
        print(f"   {'阈值':<6} {'保留类别':<8} {'保留样本':<10} {'类别占比':<10} {'样本占比':<10}")
        print("   " + "-" * 50)
        
        for threshold in thresholds:
            remaining_classes = (label_counts >= threshold).sum()
            remaining_samples = label_counts[label_counts >= threshold].sum()
            
            if remaining_classes > 0:
                class_pct = remaining_classes / unique_labels * 100
                sample_pct = remaining_samples / final_size * 100
                print(f"   >={threshold:<4} {remaining_classes:<8} {remaining_samples:<10} {class_pct:<9.1f}% {sample_pct:<9.1f}%")
        
        # 分析为什么F1这么低
        print(f"\n❗ F1分数低的原因分析:")
        very_small_classes = (label_counts <= 5).sum()
        small_classes = (label_counts <= 10).sum()
        
        print(f"   样本数≤5的类别: {very_small_classes} 个 ({very_small_classes/unique_labels*100:.1f}%)")
        print(f"   样本数≤10的类别: {small_classes} 个 ({small_classes/unique_labels*100:.1f}%)")
        
        # 建议解决方案
        print(f"\n💡 改进建议:")
        if unique_labels > 100:
            print(f"   1. 类别数过多({unique_labels}个)，建议使用阈值方法减少类别")
        if label_counts.min() < 10:
            print(f"   2. 存在样本极少的类别，建议设置最小样本数阈值")
        if label_counts.max() / label_counts.min() > 100:
            print(f"   3. 类别不平衡严重({label_counts.max()/label_counts.min():.0f}:1)，建议使用加权损失函数")
        
        return {
            'total_samples': final_size,
            'unique_labels': unique_labels,
            'label_counts': label_counts
        }
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {data_path}")
        return None
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

def main():
    """主函数"""
    print("🔍 新闻数据集标签分析工具")
    print("="*70)
    
    # 检查不同可能的数据文件路径
    possible_paths = [
        "dataset/news.csv",
        "news.csv",
        "data/news.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("❌ 找不到数据文件!")
        print("请确保以下文件之一存在:")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    # 分析数据集
    result = analyze_dataset(data_path)
    
    if result:
        print(f"\n✅ 分析完成!")
        print(f"📊 数据集包含 {result['unique_labels']} 个类别，{result['total_samples']} 个样本")
        print(f"💡 这解释了为什么Macro F1只有0.002 - 类别数太多，样本分布极不均匀!")

if __name__ == "__main__":
    main()