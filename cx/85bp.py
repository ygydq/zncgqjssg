# 导入必要的库
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 输入样本数据
X = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32,	0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1, 1.04, 1.08, 1.12, 1.16, 1.2])

# 输出样本数据
y = np.array([382.016, 382.908, 383.752, 384.595,385.388,386.137,386.814,387.48,388.059,388.651,389.095,389.589,389.988,390.469,390.857,391.308,391.736,392.099,392.389,392.658,392.954,393.153,393.309,393.413,393.503,393.594,393.696,393.783,393.864,393.958,394.051])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 归一化数据
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1,1))
X_test = scaler.transform(X_test.reshape(-1,1))

# 创建并训练BP神经网络模型
model = MLPRegressor(hidden_layer_sizes=(50,), activation='tanh', solver='lbfgs', max_iter=10000, learning_rate_init=0.01)
model.fit(X_train,y_train)

# 预测并评估模型性能
y_pred = model.predict(X_test)
mse1 =mean_squared_error(y_train,model.predict(X_train))
print('训练集MSE:', mse1)
mse = mean_squared_error(y_test,y_pred)
print('测试集MSE:', mse)


