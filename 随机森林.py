# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:23:54 2022

@author: Cyrus.L
"""

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
dataset = pd.read_excel('new_data.xlsx')
X = dataset.drop(['order_id','STATUS'],axis=1)
y = dataset['STATUS']
# dataset = pd.read_csv('woe_data2.csv')
# X = dataset.drop(['order_id','STATUS'],axis=1)
# y = dataset['STATUS']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
Y = y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = RandomForestClassifier(
        n_estimators=10, criterion='gini',
        max_depth=None,min_samples_split=2, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False, n_jobs=1, 
        random_state=None, verbose=0,
        warm_start=False, class_weight=None)  # 实例化模型RandomForestClassifier
model.fit(X_train, Y_train)  # 在训练集上训练模型
print(model)  # 输出模型RandomForestClassifier
from sklearn.metrics import roc_curve, auc
# 在测试集上测试模型
expected = Y_test  # 测试样本的期望输出
y_pred1 = model.predict(X_test)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False


y_pred1 = model.predict_proba(X_test)[:, 1]
fpr_Nb, tpr_Nb, _ = roc_curve(Y_test,y_pred1)
aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值
plt.figure(figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3)
plt.grid()
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("随机森林ROC曲线")
plt.plot(fpr_Nb, tpr_Nb, color='darkorange',
          label='ROC curve (area = %0.2f)' % aucval)
plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))


import numpy as np
ks_value = max(abs(fpr_Nb-tpr_Nb))
x = np.argwhere(abs(fpr_Nb-tpr_Nb) == ks_value)[0, 0]
plt.plot([fpr_Nb[x], fpr_Nb[x]], [fpr_Nb[x], tpr_Nb[x]], linewidth=4, color='r')
plt.text(fpr_Nb[x]+0.01,tpr_Nb[x]-0.2, 'ks='+str(format(ks_value,'0.3f')),color= 'black')
plt.xlabel('False positive', fontsize=20)
plt.ylabel('True positive', fontsize=20)
plt.title('KS_curve')
plt.show()