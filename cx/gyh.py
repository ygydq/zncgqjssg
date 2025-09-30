import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设您的数据存储在一个名为"data.csv"的文件中
data = pd.read_csv("data.csv", header=None, index_col=None)

# 按列归一化
normalized_data = data.copy()
normalized_data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:, 1:].min()) / (data.iloc[:, 1:].max() - data.iloc[:, 1:].min())

# 样本3 传感器的输出
x = normalized_data.iloc[:, 3].values.reshape(-1, 1)

# 样本3 传感器的输入
y = normalized_data.iloc[:, 0].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 创建SVR模型
svr = SVR(kernel='rbf', C=1000,  gamma=1, epsilon=0.001, max_iter=10000)

# 训练模型
svr.fit(x_train, y_train)

# 计算拟合后的输出值
y_fit = svr.predict(x_test)


# 计算拟合误差
mse = mean_squared_error(y_test, y_fit)

# 显示拟合结果
print("拟合误差: ", mse)
