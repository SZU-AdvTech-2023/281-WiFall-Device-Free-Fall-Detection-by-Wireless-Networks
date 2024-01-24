import os
import csv
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

row_data = [] #原始数据
row_integers = [] #X雏形
row_strings = [] #Y雏形
i = 0
# 文件夹路径
folder_path = 'D:\Wifall\Test'

try:
    # 获取文件夹中所有txt文件的列表
    file_list1 = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # 遍历文件列表
    for file_name in file_list1:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            # 读取整个文件内容
            file_content = file.readlines()

            # 将文本内容转为二维数组（浮点数）
            matrix = []
            for line in file_content:
                # 尝试将每个数字转为浮点数，如果失败则跳过
                try:
                    row = [float(num) for num in line.strip().split()]
                    matrix.append(row)
                except ValueError:
                    print(f"Ignoring non-numeric values in line: {line}")
            #print(len(matrix[0]))
            row_data.append(matrix)

    # 获取文件夹中所有csv文件的列表
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 遍历文件列表
    for file_name in file_list:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 打开CSV文件并读取内容
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)

            # 读取CSV文件内容
            header = None
            second_row_integers = None
            second_row_strings = None

            for row in csv_reader:
                if header is None:
                    # 保存第一行作为标题（列名）
                    header = row
                else:
                    # 保存第二行的前28个数据为整数数组
                    second_row_integers = [int(cell) for cell in row[:28]]
                    # 保存第二行的后14个数据为字符串数组
                    second_row_strings = str(row[28:])
                    break  # 只需要读取第二行
            row_integers.append(second_row_integers)
            row_strings.append(second_row_strings)
            # 打印标题、第二行整数数据和第二行字符串数据
            #print("标题:", header)
           # print("第二行的前28个整数数据:", second_row_integers)
            #print("第二行的后14个字符串数据:", second_row_strings)

except FileNotFoundError:
    print(f"文件夹 '{folder_path}' 未找到。")
except Exception as e:
    print(f"发生错误: {str(e)}")

#打印数组长度，长度都为11
#print(len(row_data))
#print(len(row_integers))
#print(len(row_strings))

#开始记录输入神经网络的数组
X_data = []
Y_data = []

#从已获取的数组中提取数据
for i in range(11):
    for j in range(14):
        Start = j
        End = j + 14
        Act = []
        #if row_integers[i][End] - row_integers[i][Start] < 800:
        for k in range(row_integers[i][Start],row_integers[i][End]):
            Act.append(row_data[i][k])
        X_data.append(Act)

# 遍历label列表
for item in row_strings:
    # 使用eval函数将字符串列表转换为实际的列表对象
    inner_list = eval(item)
    # 遍历内部列表
    for action in inner_list:
        if action == 'Walk':
            Y_data.append(0)
        if action == 'Fall':
            Y_data.append(1)
        elif action == 'Stand':
            Y_data.append(2)
        elif action == 'Sit':
            Y_data.append(3)
        #else:
            #Y_data.append(0)


#print(len(X_data))
#print(len(Y_data))

"""
maxlen = -1
ave = 0
for i in range(154):
    ave += len(X_data[i])
    if len(X_data[i]) > maxlen:
        maxlen = len(X_data[i])

print(maxlen)
print(int(ave/154))
"""

#for i in range(133):
    #print(len(X_data[i]))

# 删除最后一个元素
#del X_data[-1]

# 数据填充到固定长度
max_length = 400  # 假设的最大长度
X_data1 = pad_sequences(X_data, maxlen=max_length, padding='post', truncating='post')

# 重塑 X_data1 的形状以适应 LSTM 层
X_data1 = X_data1.reshape((X_data1.shape[0], max_length, 52))

# 转换 Y_data 为独热编码
Y_data1 = to_categorical(Y_data, num_classes=4)  # 假设有 5 个类别（包括 0）

print(len(X_data1))
print(len(Y_data1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_data1, Y_data1, test_size=0.2, random_state=43)

# 构建CNN模型
model = Sequential()
n_features = 52
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(max_length, n_features)))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 假设有3个类别
# 定义学习率衰减回调函数
lr_decay = ReduceLROnPlateau(factor=0.1, patience=1)
# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=15, batch_size=64, callbacks=[lr_decay])
#print("只区分坐")
# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试准确率: {accuracy * 100}%")

# 使用模型进行预测
y_pred = model.predict(X_test)

# 将预测结果和真实标签转换为类别标签
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# 计算误报率
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + conf_matrix.sum(axis=1) - np.diag(conf_matrix))
epsilon = 1e-7  # 为避免除以零
FPR = FP / (FP + TN + epsilon)

# 输出每个类的误报率
print("类别误报率:", FPR)

mean_FPR = np.mean(FPR)
print("平均类别误报率:", mean_FPR)



lr_decay = ReduceLROnPlateau(factor=0.1, patience=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
options = {
    "epochs": 15,
    "batch_size": 64,
    'validation_data': (X_test, y_test),
    "callbacks": [lr_decay]
}
model.fit(X_train, y_train, **options)
