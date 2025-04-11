import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ['Train', 'Dev', 'Test']
newcarinfo_count = [12428, 4099, 4214]
other_count = [15997, 5376, 5262]
newcarinfo_ratio = [43.72, 43.26, 44.47]
other_ratio = [56.28, 56.74, 55.53]

# 设置图形大小和样式
plt.figure(figsize=(12, 6))
plt.style.use('classic')

# 设置全局字体大小
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 设置柱状图的位置
x = np.arange(len(datasets))
width = 0.3  # 减小柱子宽度

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
plt.title('Small Sample Data Distribution', pad=15)
plt.xticks(x, datasets)
plt.legend(loc='upper right')

# 添加网格线使图表更清晰
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# 添加数值标签（同时显示数量和比例）
def autolabel(rects, counts, ratios):
    for i, (rect, count, ratio) in enumerate(zip(rects, counts, ratios)):
        height = rect.get_height()
        plt.annotate(f'{count:,}\n{ratio:.1f}%',  # 简化显示，移除"Count:"前缀
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 减小垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,  # 减小字体大小
                    bbox=dict(boxstyle='round,pad=0.3',  # 减小padding
                             fc='white', 
                             ec='gray', 
                             alpha=0.8))

autolabel(rects1, newcarinfo_count, newcarinfo_ratio)
autolabel(rects2, other_count, other_ratio)

# 调整y轴范围，给标签留出更多空间
plt.ylim(0, max(max(newcarinfo_ratio), max(other_ratio)) * 1.15)  # 增加15%的空间

# 调整布局以确保标签不被裁剪
plt.tight_layout()

# 保存图片
plt.savefig('small_sample_distribution.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('small_sample_distribution.png', format='png', bbox_inches='tight', dpi=300)

# 显示图形
plt.show()