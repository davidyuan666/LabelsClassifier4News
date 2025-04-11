import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['Train', 'Dev', 'Test']
newcarinfo_count = [9556, 9579, 4831]
other_count = [40444, 40421, 45169]
newcarinfo_ratio = [19.11, 19.16, 9.66]
other_ratio = [80.89, 80.84, 90.34]

# 设置图形大小和样式
plt.figure(figsize=(12, 6))
plt.style.use('classic')

# 设置柱状图的位置
x = np.arange(len(datasets))
width = 0.3

# 创建柱状图
rects1 = plt.bar(x - width/2, newcarinfo_ratio, width, 
                 label='NewCarInfo', 
                 color='#2ecc71', 
                 edgecolor='black',
                 alpha=0.8)
rects2 = plt.bar(x + width/2, other_ratio, width, 
                 label='Other', 
                 color='#3498db', 
                 edgecolor='black',
                 alpha=0.8)

# 自定义图形
plt.ylabel('Distribution Ratio (%)')
plt.title('Balanced Sample Data Distribution', pad=15)
plt.xticks(x, datasets)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# 添加数值标签
def autolabel(rects, counts, ratios):
    for i, (rect, count, ratio) in enumerate(zip(rects, counts, ratios)):
        height = rect.get_height()
        plt.annotate(f'{count:,}\n{ratio:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                             fc='white', 
                             ec='gray', 
                             alpha=0.8))

autolabel(rects1, newcarinfo_count, newcarinfo_ratio)
autolabel(rects2, other_count, other_ratio)

# 调整y轴范围，给标签留出更多空间
plt.ylim(0, max(max(newcarinfo_ratio), max(other_ratio)) * 1.15)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('balanced_sample_distribution.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('balanced_sample_distribution.png', format='png', bbox_inches='tight', dpi=300)

plt.show()