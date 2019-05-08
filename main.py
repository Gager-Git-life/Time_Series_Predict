#coding:utf-8
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#把xlsx文件转换成csv文件
def xlsx_to_csv_pd():
    print(u'[INFO]>>> 开始文件转换...')
    data_xls = pd.read_excel('test.xlsx', index_col=0)
    data_xls.to_csv('test.csv', encoding='utf-8')
    print(u'[INFO]>>> 文件转换完成!')
	
xlsx_to_csv_pd()

"""
读取文件获取训练数据和测试数据
test数据用于和最终的预测数据进行比对
"""
if(not os.path.exists('test.csv')):
	print(u'[INFO]>>> 没有发现csv文件!')
	sys.exit(0)

df = pd.read_csv('test.csv')
# 用于训练
train = df[0:400]
# 用于测试验证
test = df[400:]

# 每天为单位聚合数据集
print(u'[INFO]>>> 准备数据...')
df['time'] = pd.to_datetime(df['report_date'], format='%Y%m%d')
df.index = df['time']
df = df.resample('D').mean()
 
train['time'] = pd.to_datetime(train['report_date'], format='%Y%m%d')
train.index = train['time']
train = train.resample('D').mean()
 
test['time'] = pd.to_datetime(test['report_date'], format='%Y%m%d')
test.index = test['time']
test = test.resample('D').mean()
 
# 通过Holt-Winters季节性预测模型预测和test等量的数据
# 在图片中对比感受预测的准确性

print(u'[INFO]>>> 开始生成预测数据!')
y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['total_purchase_amt']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

y_hat_ = test.copy()
fit_ = ExponentialSmoothing(np.asarray(df['total_purchase_amt']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_['predict'] = fit_.forecast(len(test))


# matplotlib数据图像绘制
print(u'[INFO]>>> 开始绘制图像...')
plt.figure(figsize=(16, 8))
plt.plot(list(range(len(df.index))), df['total_purchase_amt'], label='Train')
plt.plot(list(range(len(train.index), len(train.index) + len(y_hat_avg.index))), y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.plot(list(range(len(df.index), len(df.index) + len(test.index))), y_hat_['predict'], label='predict')
plt.legend(loc='best')
plt.show()
