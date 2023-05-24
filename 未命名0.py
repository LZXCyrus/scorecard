# import numpy as np  
# import pandas as pd
# train= pd.read_excel('案例数据.xls')
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# # 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
# matplotlib.rcParams['axes.unicode_minus'] = False

# y = train['月运量变化量']
# p=train['年运量变化量']

# plt.hist([y,p],bins=30,label=['月运量变化量', '年运量变化量'])
# plt.legend(loc='upper left')

# plt.xlabel("运量")
# plt.ylabel("计数")
 
# 

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn
# dataset = pd.read_csv('woe_data.csv')
# #dataset = pd.read_excel('new_data.xlsx')
# X = np.asarray(dataset.drop(['order_id','STATUS'],axis=1))
# y = np.asarray(dataset['STATUS'])

# # 划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# from sklearn.linear_model import LogisticRegression
# # 模型建立
# model = LogisticRegression(solver='newton-cg',max_iter=60)
# # 拟合
# model.fit(X_train, y_train)
# # 预测测试集
# predictions = model.predict(X_test)
# print("测试集auc:",sklearn.metrics.roc_auc_score(y_test, predictions))
# # 打印准确率
# print('测试集准确率：', accuracy_score(y_test, predictions))
# dataset = pd.read_excel('dataset.xlsx')
# print(dataset.shape)
# dataset = dataset.drop_duplicates(keep='first')
# print(dataset.shape)
# dataset = dataset.T.drop_duplicates(keep='first').T
# print(dataset.shape)
# dataset.to_csv('dataset.csv')
# X = np.asarray(dataset.drop(['月价格变化量','年价格变化量','价格','价格变动','出发','到达','城市对'],axis=1))
# y = np.asarray(dataset['价格变动'])

# # 划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# from sklearn.linear_model import LogisticRegression
# # 模型建立
# model = LogisticRegression(solver='newton-cg',max_iter=50)
# # 拟合
# model.fit(X_train, y_train)
# # 预测测试集
# predictions = model.predict(X_test)
# print("测试集auc:",sklearn.metrics.roc_auc_score(y_test, predictions))
# # 打印准确率
# print('测试集准确率：', accuracy_score(y_test, predictions))
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文宋体
# plt.rcParams['axes.unicode_minus']=False #显示负号
# X = dataset.drop(['月价格变化量','年价格变化量','价格','价格变动','出发','到达','城市对'],axis=1)
# # 1、简单排序,正负分开按顺序
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# coef_LR = pd.Series(model.coef_.flatten(),index = X_test.columns,name = 'Var')
# coef_LR.to_csv('s.csv')
# plt.figure(figsize=(8,8))
# coef_LR.sort_values().plot(kind='barh')
# plt.title("Variances Importances")
# plt.savefig('fix.jpg', dpi=400)
#====================================================================

#====================================================================

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rcParams['font.family']='STsong'# 显示汉字 SimHei黑体，STsong 华文宋体还有font.style  font.size等
# plt.rcParams['axes.unicode_minus'] = False
# df = pd.read_excel('案例数据.xls')

# x= df
# plt.figure(figsize=(18,14))
# # sns.heatmap(round(x.corr('spearman'),2), cmap='coolwarm')
# # plt.title('spearman相关系数热力图')
# # plt.savefig("原始图2.svg", dpi=750, bbox_inches = 'tight')
# # plt.show()
# sns.heatmap(round(x.corr('pearson'),2), cmap='coolwarm')
# plt.title('pearson相关系数热力图')
# plt.savefig("原始图2.svg", dpi=750, bbox_inches = 'tight')
# plt.show()
# #====================================================================
# import pandas as pd
# import numpy as np
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# x = pd.read_excel('new_data.xlsx')
# x = x.drop(['order_id','STATUS'],axis=1)

# import statsmodels.api as sm
# x[np.isnan(x)] = 0
# x[np.isinf(x)] = 0
# # 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
# vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
# print(x.columns)
# print(vif)
# # for i in range(len(vif)):
# #     if vif[i]<10:
# #         vif[i]=1
# #     else:
# #         vif[i]=0
# vif = pd.DataFrame(vif)
# vif.to_csv('vif.csv')
# import pandas as pd
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import numpy as np


# # 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
# tol = [1./variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
# print(tol)
# from sklearn import datasets
# import pandas as pd
# import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
# dataset = pd.read_excel('案例数据.xls')

# X = dataset.drop(['月价格变化量','年价格变化量','价格','价格变动','出发','到达','城市对'],axis=1)

# y = dataset['价格']
# X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# b=model.tvalues
# a=model.pvalues
# b.to_csv('x.csv')



## 绘图函数库

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.rcParams['font.family']='STsong'# 显示汉字 SimHei黑体，STsong 华文宋体还有font.style  font.size等
# plt.rcParams['axes.unicode_minus'] = False
# df = pd.read_excel('dataset.xlsx')

#x= df
# plt.figure(figsize=(18,14))
# # sns.heatmap(round(x.corr('spearman'),2), cmap='coolwarm')
# # plt.title('spearman相关系数热力图')
# # plt.savefig("原始图2.svg", dpi=750, bbox_inches = 'tight')
# # plt.show()
# sns.heatmap(round(x.corr('pearson'),2), cmap='coolwarm')
# plt.title('pearson相关系数热力图')
# plt.savefig("原始图2.svg", dpi=750, bbox_inches = 'tight')
# plt.show()

# import pandas as pd
# import numpy as ny
 
# import missingno as msno
# data=pd.read_excel('dataset.xlsx')


# #'unionpayinfo_all_cost1','unionpayinfo_all_cost2','unionpayinfo_all_cost3','unionpayinfo_all_cost4','unionpayinfo_all_cost5','unionpayinfo_all_cost6','unionpayinfo_all_cost7','unionpayinfo_all_cost8','juxinliinfo_present_month_bill','juxinliinfo_last1_month_bill','juxinliinfo_last2_month_bill','juxinliinfo_last3_month_bill','juxinliinfo_last4_month_bill','juxinliinfo_last5_month_bill'
# data = data[['unionpayinfo_all_cost1','unionpayinfo_all_cost2','unionpayinfo_all_cost3','unionpayinfo_all_cost4','unionpayinfo_all_cost5','unionpayinfo_all_cost6','juxinliinfo_present_month_bill','juxinliinfo_last1_month_bill','juxinliinfo_last2_month_bill','juxinliinfo_last3_month_bill','juxinliinfo_last4_month_bill','juxinliinfo_last5_month_bill'
#              ]]
# ## 绘制缺失值矩阵图
# data = data.replace(-99999, np.nan)
# msno.matrix(data,labels=True)
# plt.savefig("原始图2.svg", dpi=750, bbox_inches = 'tight')
# import numpy as np
# import pandas as pd


# from sklearn.model_selection import train_test_split
# from sklearn.inspection import permutation_importance
# from matplotlib import pyplot as plt
# import seaborn as sns
# from sklearn.datasets import make_regression
# from xgboost import XGBRegressor
# from matplotlib import pyplot
# from xgboost import XGBRFClassifier
# import lightgbm as lgb
# x = pd.read_excel('dataset.xlsx')
# print(x.head(0))
# y= x['STATUS']
# X = x.drop(['order_id','STATUS'],axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
# # xgb = XGBRFClassifier(n_estimaters = 1000)
# xgb = lgb.LGBMClassifier(n_estimaters = 1000)
# xgb.fit(X_train, y_train)
# importance = xgb.feature_importances_
# s = pd.DataFrame(importance)
# s.to_csv('xxx.csv')
# # summarize feature importance
# for i,v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X = pd.read_csv('woe_data2.csv')
y=X['STATUS']
x = X.drop(['STATUS','order_id'],axis=1)


# model = sm.Logit(y, X).fit()
# b=model.tvalues
# a=model.pvalues
#x = x.drop(['order_id','STATUS'],axis=1)

import statsmodels.api as sm
x[np.isnan(x)] = 0
x[np.isinf(x)] = 0
# 当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
vif = [variance_inflation_factor(x.values, x.columns.get_loc(i)) for i in x.columns]
print(x.columns)
print(vif)
# for i in range(len(vif)):
#     if vif[i]<10:
#         vif[i]=1
#     else:
#         vif[i]=0
vif = pd.DataFrame(vif)
vif.to_csv('vif.csv')