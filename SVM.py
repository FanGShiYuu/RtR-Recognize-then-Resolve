import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 仅av左转90% 仅av直行99% 左转+直行87%
# 读取数据
# av data
data1 = pd.read_csv('argo_data/interaction_av_lt/left_turn_rush_output.csv')
data2 = pd.read_csv('argo_data/interaction_av_lt/left_turn_yield_output.csv')
data3 = pd.read_csv('argo_data/interaction_av_gs/left_turn_rush_output.csv')
data4 = pd.read_csv('argo_data/interaction_av_gs/left_turn_yield_output.csv')
merged_data_av_lt = pd.concat([data1, data2])
merged_data_av_gs = pd.concat([data3, data4])
merged_data_all = pd.concat([merged_data_av_lt, merged_data_av_gs])

# hv data
# data1 = pd.read_csv('argo_data/interaction_hv/left_turn_rush_output.csv')
# data2 = pd.read_csv('argo_data/interaction_hv/left_turn_yield_output.csv')
# merged_data_all = pd.concat([data1, data2])

for i, merged_data in enumerate([merged_data_av_lt, merged_data_av_gs, merged_data_all]):
    if i == 0:
        data_type = 'av_lt'
    elif i == 1:
        data_type = 'av_gs'
    else:
        data_type = 'all'

# for merged_data in [merged_data_all]:
#     data_type = 'all(hv)'

    # 提取特征和标签
    # X = merged_data[['ego_ttcp', 'agent_ttcp', 'agent_a_c']]  # , 'ego_dis2des', 'agent_dis2des', 'ego_current_velocity', 'agent_current_velocity'
    # y = merged_data['av_pass_first']
    #
    # # 将True/False转换为二进制值（SVM需要数值类型的标签）
    # y = y.map({True: 1, False: 0})

    # 以HV为主视角计算其权重
    X = merged_data[['agent_ttcp', 'ego_ttcp', 'ego_a_c']]  # , 'ego_dis2des', 'agent_dis2des', 'ego_current_velocity', 'agent_current_velocity'
    y = merged_data['av_pass_first']

    # 将True/False转换为二进制值（SVM需要数值类型的标签）
    y = y.map({True: 0, False: 1})

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print('训练样本个数', len(X_train))

    # 对特征进行标准化
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # 创建并训练SVM模型
    print('train start')
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    print('train finish')

    w = svm_model.coef_[0]  # 权重向量
    b = svm_model.intercept_[0]  # 偏置项

    print("Weight (w):", w)
    print("Bias (b):", b)

    # 训练好的模型 svm_model
    # 保存模型到文件
    # joblib.dump(svm_model, f'./svm_models/svm_model_{data_type}.pkl')

    # # 从文件加载模型
    # svm_model = joblib.load('svm_model.pkl')
    #
    # # 使用加载的模型进行预测
    # y_pred = svm_model.predict(X_test)

    # 进行预测
    y_pred = svm_model.predict(X_test)

    # 计算模型的准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model {data_type} Accuracy: {accuracy:.2f}')

    # 创建一个新的DataFrame存储测试数据及其预测结果
    test_data = X_test.copy()
    test_data['ego_ttcp'] = X_test['ego_ttcp']  # 保留原始的 ego_ttcp
    test_data['y_true'] = y_test.values
    test_data['y_pred'] = y_pred

    # 定义区间
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '>10']
    # 将 ego_ttcp 分成不同区间
    test_data['ttcp_bin'] = pd.cut(test_data['ego_ttcp'], bins=bins, labels=labels)
    # 计算每个区间的准确率
    accuracy_by_bin = test_data.groupby('ttcp_bin').apply(
        lambda x: pd.Series({
            'accuracy': accuracy_score(x['y_true'], x['y_pred']),
            'sample_count': x.shape[0]
        })
    )
    # 打印每个区间的准确率
    print('ttcp准确率', accuracy_by_bin)


    # def compute_confidence_interval(data, confidence=0.95):
    #     n = len(data)
    #     mean = np.mean(data)
    #     std_error = stats.sem(data)
    #     h = std_error * stats.t.ppf((1 + confidence) / 2., n - 1)
    #     return mean - h, mean + h
    #
    #
    # # 计算每个区间的准确率及置信区间
    # for label in test_data['ttcp_bin'].unique():
    #     group_data = test_data[test_data['ttcp_bin'] == label]['y_pred']
    #     ci = compute_confidence_interval(group_data)
    #     print(f"区间 {label} 置信区间: {ci}")

    # # 定义区间
    # bins = [-float('inf'), -6, -4, -2, 0, 2, 4, 6, 8, float('inf')]
    # labels = ['<-6', '-6--4', '-4--2', '-2-0', '0-2', '2-4', '4-6', '6-8', '>8']
    # # 将 ego_ttcp 分成不同区间
    # test_data['ttcp_bin'] = pd.cut(test_data['ego_ttcp'] - test_data['agent_ttcp'], bins=bins, labels=labels)
    # # 计算每个区间的准确率
    # accuracy_by_bin = test_data.groupby('ttcp_bin').apply(
    #     lambda x: pd.Series({
    #         'accuracy': accuracy_score(x['y_true'], x['y_pred']),
    #         'sample_count': x.shape[0]
    #     })
    # )
    # # 打印每个区间的准确率
    # print('Δttcp准确率', accuracy_by_bin)

    # 绘制柱状图
    accuracy_by_bin.plot(kind='bar', color='skyblue', edgecolor='black')
    # plt.title('Accuracy by ego_ttcp range')
    # plt.xlabel('ego_ttcp range')
    # plt.ylabel('Accuracy')
    # plt.xticks(rotation=0)
    # plt.ylim(0, 1)
    # plt.show()



