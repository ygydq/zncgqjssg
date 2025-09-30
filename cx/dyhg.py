import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载Excel文件
df = pd.read_excel('dyhgsj.xlsx')

# 从DataFrame中提取U_P、U_T、T和P值
U_P = df[df.iloc[:, 1] == 'U_P'].iloc[:, 2:].values.flatten()
U_T = df[df.iloc[:, 1] == 'U_T'].iloc[:, 2:].values.flatten()
T = df[df.iloc[:, 1] == 'U_P'].iloc[:, 0].values.repeat(df.columns[2:].shape[0])
P = np.tile(df.columns[2:].astype(float), df[df.iloc[:, 1] == 'U_P'].shape[0])

# 将U_P、U_T和T组合成回归的输入特征
X = np.column_stack((U_P, U_T))

# 创建degree=2的多项式特征
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 创建并训练模型
model = LinearRegression()
model.fit(X_poly, P)

# 打印模型参数
print("截距: ", model.intercept_)
print("系数: ", model.coef_)

# 预测结果
y_pred = model.predict(X_poly)
print("预测结果: ", y_pred)

# 计算工作温度变化范围
delta_T = T.max() - T.min()

# 计算压力传感器满量程输出值和输入值
U_FS = max(U_P)
P_FS = max(P)

# 计算工作温度变化ΔT范围内，压力传感器零点漂移最大值和零点压力最大偏差
zero_pressure_column = df.columns[df.columns == 0]
U_P_at_zero_P = df[df.iloc[:, 1] == 'U_P'].loc[:, zero_pressure_column].values.flatten()
delta_U_0m=abs(min(U_P_at_zero_P) - max(U_P_at_zero_P))

# 计算逆模型融合计算在ΔT范围内的零点压力最大偏差
zero_pressure_indices = np.where(P == min(P))
P_pred_at_zero_P = y_pred[zero_pressure_indices] # 压力为0时的预测压力值
delta_P_0m = max(abs(P_pred_at_zero_P - 0))  # 预测值与给定值0的差值的最大值

# 计算零位温度系数
alpha_0_before = delta_U_0m / U_FS * 1 / delta_T
alpha_0_after = delta_P_0m / P_FS * 1 / delta_T

# 找到满量程压力对应的索引
full_scale_pressure_indices = np.where(P == P_FS)

# 获取满量程压力对应的压力传感器输出值
U_P_at_full_scale_P = U_P[full_scale_pressure_indices]

# 获取所有的唯一温度值
unique_temperatures = np.unique(T)

# 找到最小温度对应的索引
min_temperature_index = np.argmin(unique_temperatures)

# 获取最小温度下的压力传感器输出值
U_P_at_min_P=U_P_at_full_scale_P[min_temperature_index]

# 找到最大温度对应的索引
max_temperature_index = np.argmax(unique_temperatures)

# 获取最大温度下的压力传感器输出值
U_P_at_max_P=U_P_at_full_scale_P[max_temperature_index]

# 获取满量程压力对应的预测压力值
P_pred=y_pred[full_scale_pressure_indices]

# 获取最小温度下的预测压力值
P_pred_at_min_P=P_pred[min_temperature_index]

# 获取最大温度下的预测压力值
P_pred_at_max_P=P_pred[max_temperature_index]


# 计算灵敏度温度系数
alpha_s_before = (U_P_at_min_P - U_P_at_max_P) / (U_P_at_min_P * delta_T)
alpha_s_after = (P_pred_at_min_P - P_pred_at_max_P) / (P_pred_at_min_P * delta_T)

print("零位温度系数（融合处理之前）: ", alpha_0_before)
print("零位温度系数（融合处理之后）: ", alpha_0_after)
print("灵敏度温度系数（融合处理之前）: ", alpha_s_before)
print("灵敏度温度系数（融合处理之后）: ", alpha_s_after)


# 计算均方误差
mse = mean_squared_error(P, y_pred)
print("均方误差: ", mse)

# 计算均方根误差
rmse = np.sqrt(mse)
print("均方根误差: ", rmse)

# 计算平均绝对误差
mae = mean_absolute_error(P, y_pred)
print("平均绝对误差: ", mae)

# 计算误差平方和
sse = np.sum((P - y_pred) ** 2)
print("误差平方和: ", sse)

# 计算平均绝对百分比误差
mape = np.mean(np.abs((P - y_pred) / P)) * 100
print("平均绝对百分比误差: ", mape)

# 计算决定系数
r2 = r2_score(P, y_pred)
print("决定系数: ", r2)

# 计算校正决定系数
n = P.shape[0]  # 样本数量
p = X_poly.shape[1]  # 特征数量
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("校正决定系数: ", adjusted_r2)

