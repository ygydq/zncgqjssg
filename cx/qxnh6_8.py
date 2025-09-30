# 引用第三方库
import numpy as np

# 定义输入样本数据
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])   
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

# 使用numpy的polyfit函数进行数据拟合，得到多项式的系数
coefficients = np.polyfit(x, y, 3) 

# 使用numpy的poly1d函数生成多项式方程
polynomial_equation = np.poly1d(coefficients)

# 使用numpy的polyval函数进行预测
y1= np.polyval(coefficients, 3.5)  
                 
# 输出显示多项式方程和预测结果
print("多项式方程为：",polynomial_equation)
print("预测结果为：",y1)
