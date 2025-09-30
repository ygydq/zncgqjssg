#引用第三方库
import numpy as np
from sklearn import linear_model

#将表中测试数据存入python变量
x=np.array([0, 1, 2, 3, 4, 5])
y01=np.array([0, 4, 9, 14, 20, 25])
y11=np.array([0, 5, 10, 16, 21, 25])
y02=np.array([0, 5, 10, 15, 19, 25])
y12=np.array([0, 5, 11, 16, 20, 25])

#求取输出平均值
y=(y01+y11+y02+y12)/4

#求取端点坐标
xx=np.array([x.min(), x.max()])
yy=np.array([y.min(), y.max()])

#求取拟合直线
clf = linear_model.LinearRegression()
clf.fit(xx.reshape(-1, 1),yy)
k=clf.coef_[0]
b=clf.intercept_
y_pre=clf.predict(x.reshape(-1, 1))

#打印拟合直线的斜率和截距
print("拟合直线的斜率: ", k)
print("拟合直线的截距: ", b)

#计算性能指标
deltay_max=np.array([abs(y_pre-y01),abs(y_pre-y02), \
abs(y_pre-y11),abs(y_pre-y12)]).max()
xxd=deltay_max/max(y)
lmd=(y.max()-y.min())/(x.max()-x.min())
deltah_max=np.array([abs(y11-y01),abs(y12-y02)]).max()
cz=deltah_max/max(y)
cf_max=np.array([abs(y02-y01),abs(y12-y11)]).max()
cfwc=cf_max/max(y)

#打印性能指标
print("线性度: ", xxd)
print("灵敏度: ", lmd)
print("迟滞: ", cz)
print("重复性误差: ", cfwc)
