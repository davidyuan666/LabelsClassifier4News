import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['NewCarAnalysis', 'NewCarPrice', 'NewCarLaunch', 'ReleaseAppearance', 
          'Pre-heating', 'Pre-sale', 'ConfigExposure', 'SpyPhotos',
          'RealCarExposure', 'DeclarationImages', 'OfficialImages', 'NewCarArrival']

sizes = [150277, 67806, 65213, 37196, 28486, 21557, 10197, 8354, 5487, 4966, 3916, 1441]

# 计算百分比
total = sum(sizes)
proportions = [size/total*100 for size in sizes]

# 设置颜色
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c',
          '#e67e22', '#34495e', '#7f8c8d', '#c0392b', '#16a085', '#d35400']

# 设置图形大小
plt.figure(figsize=(12, 8))

# 创建饼图
# 将小于3%的部分突出显示
explode = [0.05 if p < 3 else 0 for p in proportions]

# 绘制饼图
patches, texts, autotexts = plt.pie(sizes, 
                                  explode=explode,
                                  labels=labels,
                                  colors=colors,
                                  autopct=lambda pct: f'{pct:.1f}%\n({int(pct*total/100):,})',
                                  pctdistance=0.85,
                                  labeldistance=1.1)

# 设置标签文本大小
plt.setp(autotexts, size=8)
plt.setp(texts, size=9)

# 添加标题
plt.title('Data Distribution of News Labels', pad=20, size=14)

# 确保饼图是圆形的
plt.axis('equal')

# 添加图例
plt.legend(labels, 
          title="Labels",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# 调整布局以确保所有元素可见
plt.tight_layout()

# 保存图片
plt.savefig('data_distribution_pie.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('data_distribution_pie.png', format='png', bbox_inches='tight', dpi=300)

# 显示图形
plt.show()