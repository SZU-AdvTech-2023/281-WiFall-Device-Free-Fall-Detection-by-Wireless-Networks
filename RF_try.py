import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# 读取数据
res = pd.read_csv('svm_output.csv')

# 数据预处理
num_class = len(res.iloc[:, -1].unique())
#print("只区分坐:", num_class)
num_res = res.shape[0]
num_size = 0.75
res = shuffle(res, random_state=42)
P_train, P_test = [], []
T_train, T_test = [], []
for i in range(1, num_class + 1):
    mid_res = res[res.iloc[:, -1] == i]
    mid_size = mid_res.shape[0]
    mid_train = round(num_size * mid_size)
    P_train.extend(mid_res.iloc[:mid_train, :-1].values.tolist())
    T_train.extend(mid_res.iloc[:mid_train, -1].values.tolist())
    P_test.extend(mid_res.iloc[mid_train:, :-1].values.tolist())  # 修正测试集数据
    T_test.extend(mid_res.iloc[mid_train:, -1].values.tolist())   # 修正测试集数据

P_train, T_train = np.array(P_train), np.array(T_train)
P_test, T_test = np.array(P_test), np.array(T_test)
M, N = P_train.shape[0], P_test.shape[0]

# 数据缩放
scaler = MinMaxScaler()
dataset = np.vstack((P_train, P_test))
dataset_scale = scaler.fit_transform(dataset)
P_train = dataset_scale[:M, :]
P_test = dataset_scale[M:(M + N), :]

# 随机森林参数网格
param_grid = {
    'n_estimators': [5, 10, 25, 50, 100, 200, 250],
    'max_features': ['sqrt', 'log2'],  # Removed 'auto', as it's not valid for RandomForestClassifier
    'max_depth': [None, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}

# 创建随机森林分类器
rf_classifier = RandomForestClassifier()

# 网格搜索
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(P_train, T_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数的模型
best_rf_classifier = grid_search.best_estimator_

# 测试集预测
predictions = best_rf_classifier.predict(P_test)
accuracy = accuracy_score(T_test, predictions)

# 输出测试集准确率
print("测试集准确率:", accuracy)

# 测试集预测
predictions = best_rf_classifier.predict(P_test)

# 计算混淆矩阵
conf_matrix = confusion_matrix(T_test, predictions)

# 计算误报率
# 对于多类分类，我们需要计算每个类的误报率，然后可以取平均值
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + np.diag(conf_matrix))

# 为了避免除以零，我们可以添加一个小的值 epsilon
epsilon = 1e-7
FPR = FP / (FP + TN + epsilon)

# 输出每个类的误报率
print("类别误报率:", FPR)

# 如果需要，还可以计算平均误报率
mean_FPR = np.mean(FPR)
print("平均误报率:", mean_FPR)

mean_FPR = np.mean(FPR)
print("平均类别误报率:", mean_FPR)