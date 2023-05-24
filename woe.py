from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
#woe编码代码

def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -99999) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=5,  # 最大叶子节点数
                                 min_samples_leaf=0.06)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x+10]

    return boundary


def feature_woe_iv(x: pd.Series, y: pd.Series, nan: float = -99999) -> pd.DataFrame:
    '''
        计算变量各个分箱的WOE、IV值，返回一个DataFrame
    '''
    x = x.fillna(nan)
    boundary = optimal_binning_boundary(x, y, nan)  # 获得最优分箱边界值列表
    df = pd.concat([x, y], axis=1)  # 合并x、y为一个DataFrame，方便后续计算
    df.columns = ['x', 'y']  # 特征变量、目标变量字段的重命名
    df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间

    grouped = df.groupby('bins')['y']  # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])
    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()  # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()  # 坏客户占比
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()  # 总客户占比

    result_df['bad_rate'] = result_df['bad'] / result_df['total']  # 坏比率

    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])  # WOE
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV

    print(f"该变量IV = {result_df['iv'].sum()}")

    return boundary, result_df
x = pd.read_excel('dataset.xlsx')
y= x['STATUS']
x = x.drop(['order_id','STATUS'],axis=1)
head = x.columns.values.tolist()
woe = []
iv = []
for i in range(len(head)):
    print(head[i])
    m,n = feature_woe_iv(x[head[i]],y)
#    print(n['woe'][1])
    print(n)
    woe.append([n['woe'].sum()][0])
    iv.append([n['iv'].sum()][0])

import csv
header = ['cust_id','woe','iv']
s=[]
for i in range(len(head)):
    s.append([head[i],woe[i],iv[i]])
# print(s)
# data = list(map(lambda x:[x],ans))

with open('s.csv', 'w', encoding='UTF8',newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    for i in s:
    # write the data
        writer.writerow(i)
        
