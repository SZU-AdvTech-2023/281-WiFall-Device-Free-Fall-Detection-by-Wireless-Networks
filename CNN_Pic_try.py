import csv
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

"""
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
                    #print(len(row))
                    matrix.append(row)
                except ValueError:
                    print(f"Ignoring non-numeric values in line: {line}")
            #print(matrix)
            row_data.append(matrix)
            #print(len(row_data))

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
#print(len(row_data))

#开始记录输入神经网络的数组
X_data = []
Y_data = []

# 遍历label列表
for item in row_strings:
    # 使用eval函数将字符串列表转换为实际的列表对象
    inner_list = eval(item)
    # 遍历内部列表
    for action in inner_list:
        if action == 'Walk':
            Y_data.append(0)
        elif action == 'Fall':
            Y_data.append(1)
        elif action == 'Stand':
            Y_data.append(2)
        elif action == 'Sit':
            Y_data.append(3)

data_dir = 'D:\Wifall\PIC'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for file_index in range(len(row_data)):
    for j in range(14):
        Start = j
        End = j + 14
        Act = []
        for k in range(row_integers[file_index][Start], row_integers[file_index][End]):
            Act.append(row_data[file_index][k])
        # 将矩阵数据转换为图像格式
        reshaped_data = np.resize(Act, (300, 52))
        normalized_data = (reshaped_data - np.min(reshaped_data)) * (255 / (np.max(reshaped_data) - np.min(reshaped_data)))
        image = np.reshape(normalized_data,(300, 52))
        # 将图像的深度转换为CV_8U类型
        image = cv2.convertScaleAbs(image)

        # 将图像转换为3通道图像
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 将图像转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 进行直方图均衡化
        equalized_image = cv2.equalizeHist(gray_image)

        # 进行自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)

        # 保存灰度图像
        file_name = f'{data_dir}/file_{file_index}_action_{Y_data[j]}_{j}.jpg'
        cv2.imwrite(file_name, gray_image)

        print(f'已保存：{file_name}')


def create_label_mapping(image_folder):
    labels = set()
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".jpg"):  # 假设图像是 PNG 格式
            label = img_name[14]  # 提取第 15 个字符
            labels.add(label)
    return {label: idx for idx, label in enumerate(labels)}

def load_dataset(image_folder, label_mapping, image_size=(300, 52)):
    images = []
    labels = []
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".jpg"):  # 假设图像是 PNG 格式
            img_path = os.path.join(image_folder, img_name)
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            images.append(img)

            label = img_name[14]  # 提取第 15 个字符
            labels.append(label_mapping[label])  # 使用映射转换标签

    return np.array(images), np.array(labels)
"""

# 载入数据集
image_folder = 'D:\Wifall\PIC'
label_mapping = create_label_mapping(image_folder)
images, labels = load_dataset(image_folder, label_mapping)

# 划分数据集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 将标签转换为 one-hot 编码
num_classes = len(np.unique(train_labels))
print(num_classes)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# 构建 CNN 模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 52, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=15, validation_split=0.2)

# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"测试准确率: {test_acc}")
