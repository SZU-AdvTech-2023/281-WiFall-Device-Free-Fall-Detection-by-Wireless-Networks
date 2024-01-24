from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ...您之前的函数 create_label_mapping 和 load_dataset ...

def create_label_mapping(image_folder):
    labels = set()
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".png"):  # 假设图像是 PNG 格式
            label = img_name[14]  # 提取第 15 个字符
            labels.add(label)
    return {label: idx for idx, label in enumerate(labels)}

def load_dataset(image_folder, label_mapping, image_size=(150, 52)):
    images = []
    labels = []
    for img_name in os.listdir(image_folder):
        if img_name.endswith(".png"):  # 假设图像是 PNG 格式
            img_path = os.path.join(image_folder, img_name)
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            images.append(img)

            label = img_name[14]  # 提取第 15 个字符
            labels.append(label_mapping[label])  # 使用映射转换标签

    return np.array(images), np.array(labels)
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
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# 重塑图像以适应 LSTM 输入 (这里取决于你的数据和 LSTM 结构)
# 假设我们将每个图像的每一行作为一个时间步骤
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], -1)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], -1)

# 构建 LSTM 模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(train_images.shape[1], train_images.shape[2])),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=15, validation_split=0.2)

# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"测试准确率: {test_acc}")
