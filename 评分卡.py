import pandas as pd
import matplotlib.pyplot as plt
from scorecardbundle.feature_discretization import ChiMerge as cm  # ChiMerge特征离散
from scorecardbundle.feature_encoding import WOE as woe  # WOE编码实现
from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc  # 模型训练-逻辑回归
from scorecardbundle.model_evaluation import ModelEvaluation as me  # 模型评估
from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia

# 01读取数据
def read_csv():
    bd_data = pd.read_excel('new_data.xlsx')
    bd_data = bd_data.set_index('bd_code')  # 设置bd_code索引
    # 将object转化为float
    col = list(bd_data.columns)
    bd_data[col] = bd_data[col].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # 获取关键字表
    bd_data = bd_data[bd_data['con_num'] > 5]  # 合同数小于0的BD不参与评分
    bd_data = bd_data[['amount_char_rate', 'loss_num_rate', 'loss_rate']]

    # 归一化
    bd_data = normalized(bd_data, 'amount_char_rate')  # 归一化
    bd_data = normalized(bd_data, 'loss_num_rate')  # 归一化
    bd_data = normalized(bd_data, 'loss_rate')  # 归一化
    bd_data.to_csv('01归一化后的样本集.csv', header=True, index=True)
    return bd_data


# 归一化
def normalized(X, feature_name):
    max_x = X[feature_name].max()
    min_x = X[feature_name].min()
    X[feature_name] = X[feature_name].apply(lambda x: (x - min_x) / (max_x - min_x))
    return X


def mark_score(train_data, column, flag):
    train_data[column + '_num'] = train_data[column].rank(ascending=flag, method='dense')
    max_num = max(train_data[column + '_num'])
    train_data[column + '_num'] = train_data[column + '_num'] / max_num * 100
    return train_data



# 03 样本标注
def feature_goal(dataset):
    dataset['score_num'] = dataset['amount_char_rate'] * 0.5 + dataset[
        'loss_num_rate'] * 0.25 + dataset['loss_rate'] * 0.25

    q95 = dataset.score_num.quantile(0.95)
    q05 = dataset.score_num.quantile(0.05)
    # 截尾，避免离群值对数据造成影响
    dataset = dataset.loc[lambda x: x['score_num'] > q05]
    dataset = dataset.loc[lambda x: x['score_num'] < q95]

    # 平均值
    truncated_average = dataset.score_num.quantile(0.5)
    dataset.loc[dataset['score_num'] >= truncated_average, 'score_num'] = 1
    dataset.loc[dataset['score_num'] < truncated_average, 'score_num'] = 0

    dataset.rename(columns={'score_num': 'tag'}, inplace=True)
    dataset.to_csv('02标注后的样本集.csv', header=True, index=True)

    # 获取训练集
    train_data = dataset.sample(frac=0.75, random_state=0)
    # 获取测试集
    test_data = dataset[~dataset.index.isin(train_data.index)]

    train_data.to_csv('03训练集.csv', header=True, index=True)
    test_data.to_csv('04测试集.csv', header=True, index=True)
    # 拆分特征和标签
    train_X, train_y = train_data[['amount_char_rate', 'loss_num_rate', 'loss_rate']], train_data['tag']
    test_X, test_y = test_data[['amount_char_rate', 'loss_num_rate', 'loss_rate']], test_data['tag']
    X, y = dataset[['amount_char_rate', 'loss_num_rate', 'loss_rate']], dataset['tag']
    return train_X, train_y, test_X, test_y, X, y


# 04特征离散化（基于ChiMerge）分箱
def ChiMerge(train_X, train_y):
    trans_cm = cm.ChiMerge(max_intervals=6, min_intervals=2, output_dataframe=True)
    result_cm = trans_cm.fit_transform(train_X, train_y)
    for i in range(0,8):
        col = train_X.columns[i]

        fia.plot_event_dist(result_cm[col],train_y,x_rotation=60)
    return result_cm


# 05特征编码(基于证据权重WOE)
def woe_fun(result_cm, train_y):
    trans_woe = woe.WOE_Encoder(output_dataframe=True)
    result_woe = trans_woe.fit_transform(result_cm, train_y)  # WOE运行很快，此任务仅需1秒
    return trans_woe, result_woe


# 06模型训练
def model_train(trans_woe, result_woe, train_X, train_y):
    model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-5, basePoints=60, verbose=True)
    model.fit(result_woe, train_y)
    model.woe_df_.to_csv(r'05模型详情.csv', header=True, index=False)
    return model


def predict_result(model, X):
    result = model.predict(X)  # 得出训练集的结果分数
    result.index = X.index  # 使结果对应BD号
    result.to_csv(r'06预测结果.csv', header=True, index=True)
    return result


# 08模型评估
def model_evaluation(y, result):
    evaluation = me.BinaryTargets(y, result['TotalScore'])
    print("模型评估结果：")
    print(evaluation.ks_stat())
    print(evaluation.plot_all())


# 09分数校正
def correction_score(result_score):
    min_score = min(result_score['TotalScore'])
    max_score = max(result_score['TotalScore'])

    print("#####模型分数概况：######")
    print('最小值:' + str(min_score))
    print('最大值:' + str(max_score))
    print('平均值:' + str(result_score['TotalScore'].mean()))
    print('中位数:' + str(result_score['TotalScore'].median()))

    q5 = result_score.TotalScore.quantile(0.5)
    q7 = result_score.TotalScore.quantile(0.7)
    q9 = result_score.TotalScore.quantile(0.9)

    # D:70以下  C:70-80  B:80-90  A:90-100
    result_score['level'] = result_score['TotalScore'].apply(lambda x: get_level(x, q5, q7, q9))
    result_score.to_csv(r'07划分等级后的结果.csv', header=True, index=True)

# 等级划分函数
def get_level(score, q5, q7, q9):
    if score > q9:
        return 'A'
    elif score > q7:
        return 'B'
    elif score > q5:
        return 'C'
    else:
        return 'D'


# 数据结果分布展示
def display(data_df):
    data_df.TotalScore.hist(bins=50)
    # 构建图像
    plt.ylabel('样本数量')
    plt.xlabel('分数')
    plt.show()

from sklearn.model_selection import train_test_split
# 主程序入口
if __name__ == '__main__':
    dataset = pd.read_excel('new_data.xlsx')
    X = dataset.drop(['order_id','STATUS'],axis=1)
    y = dataset['STATUS']
    
    # 划分为训练集和测试集
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    result_cm = ChiMerge(train_X, train_y)
    print(result_cm)
    

    # 计算woe
    trans_woe, result_woe = woe_fun(result_cm, train_y)
    # 训练模型
    model = model_train(trans_woe, result_woe, train_X, train_y)
    result_woe.to_csv('woe_data.csv')

    # 预测训练集
    train_result = predict_result(model, train_X)
    # 训练集评估
    model_evaluation(train_y, train_result)
    # 预测测试集
    test_result = predict_result(model, test_X)
    # 测试集评估
    model_evaluation(test_y, test_result)
    # 预测总体
    X_result = predict_result(model, X)
    # 分数简单统计 等级划分
    correction_score(X_result)
    display(X_result)