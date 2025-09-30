import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 绘图显示中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取CSV文件，指定无表头且仅读取前两列
df = pd.read_csv('wcsj.csv', header=None,skiprows=1, usecols=[0, 1], names=['x', 'error'])

# 2. 提取数据并计算最大误差点
x = df['x'].values
y = df['error'].values
max_idx = np.argmax(y)  # 获取最大误差索引
max_x, max_y = x[max_idx], y[max_idx]

# 3. 创建画布与坐标轴
plt.figure(figsize=(10, 6), dpi=600)
plt.plot(x, y, 'k-', linewidth=1.5)  # 绘制误差曲线
plt.scatter(max_x, max_y, color='black', s=100, zorder=5)  # 标记最大点

# 4. 标注最大误差点（箭头+文字）
plt.annotate(f'最大误差: {max_y:.4f}', 
             xy=(max_x, max_y), 
             xytext=(max_x+0.1*(max(x)-min(x)), max_y*0.9),  # 偏移避免重叠
             arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1.5),
             fontsize=12)
             #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

# 5. 设置图表格式
#plt.title('误差分布图', fontsize=16, fontweight='bold')
plt.xlabel('输入值', fontsize=14)
plt.ylabel('误差', fontsize=14)
plt.grid(linestyle='--', alpha=0.6)  # 虚线网格
#plt.legend(loc='upper right', fontsize=12)
plt.tick_params(axis='both', labelsize=12)  # 刻度标签字体

# 6. 保存并显示
plt.tight_layout()
plt.savefig('error_plot.jpg', bbox_inches='tight')  # 保存为高分辨率图片
plt.show()
