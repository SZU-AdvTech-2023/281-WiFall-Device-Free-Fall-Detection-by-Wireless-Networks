import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# 导入数据
res = pd.read_csv('svm_output.csv')

# 分析数据
num_class = len(res.iloc[:, -1].unique())  # 类别数
#print("只区分站:", num_class)
num_res = res.shape[0]                   # 样本数
num_size = 0.75                           # 训练集占数据集的比例
res = shuffle(res, random_state=42)      # 打乱数据集

# 设置变量存储数据
P_train, P_test = [], []
T_train, T_test = [], []

# 划分数据集
for i in range(1, num_class + 1):
    mid_res = res[res.iloc[:, -1] == i]   # 循环取出不同类别的样本
    mid_size = mid_res.shape[0]            # 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size)  # 得到该类别的训练样本个数

    P_train.extend(mid_res.iloc[:mid_tiran, :-1].values.tolist())  # 训练集输入
    T_train.extend(mid_res.iloc[:mid_tiran, -1].values.tolist())   # 训练集输出

    P_test.extend(mid_res.iloc[:mid_tiran, :-1].values.tolist())  # 测试集输入
    T_test.extend(mid_res.iloc[:mid_tiran, -1].values.tolist())    # 测试集输出

# 转换为 NumPy 数组
P_train, T_train = np.array(P_train), np.array(T_train)
P_test, T_test = np.array(P_test), np.array(T_test)

# 得到训练集和测试样本个数
M, N = P_train.shape[0], P_test.shape[0]

# 数据预处理，将训练集和测试集归一化到 [0,1] 区间
scaler = MinMaxScaler()
dataset = np.vstack((P_train, P_test))
dataset_scale = scaler.fit_transform(dataset)

P_train = dataset_scale[:M, :]
P_test = dataset_scale[M:(M + N), :]

# 定义参数网格
param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'gamma': ['auto', 'scale', 0.1, 1, 10]}

# 创建SVC实例
svm_classifier = SVC()

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(P_train, T_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_svm_classifier = grid_search.best_estimator_
predictions = best_svm_classifier.predict(P_test)

# 计算准确率
accuracy = accuracy_score(T_test, predictions)
print("测试集准确率:", accuracy)

# 使用最佳模型进行预测
predictions = best_svm_classifier.predict(P_test)

# 计算混淆矩阵
conf_matrix = confusion_matrix(T_test, predictions)

# 计算误报率
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + conf_matrix.sum(axis=1) - np.diag(conf_matrix))
epsilon = 1e-7  # 为避免除以零
FPR = FP / (FP + TN + epsilon)

# 输出每个类的误报率
print("类别误报率:", FPR)

mean_FPR = np.mean(FPR)
print("平均类别误报率:", mean_FPR)