import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class SolarDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((300, 52))
                img_array = np.array(img)
                images.append(img_array)
                labels.append(int(label))
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    return images, labels

def get_data(data_dir, test_size=0.2, random_state=42):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, y_train), (X_test, y_test)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # 前两层
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三层
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 后续四层卷积层（具体细节需要根据实际需求调整）
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 此处省略其他卷积层的定义...

        # Dropout层
        self.drop_out = nn.Dropout()

        # 全连接层
        self.fc1 = nn.Linear(128 * X * Y, 256)  # X, Y 需要根据前面层的输出尺寸计算得出
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # 此处省略其他卷积层的前向传播...
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            labels = labels.long()  # 转换为 Long 类型
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.long()  # 转换为 Long 类型
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, train_accuracies, test_losses, test_accuracies


def main():
    pic_dir_data = 'D:\Wifall\PIC'
    (X_train, y_train), (X_test, y_test) = get_data(pic_dir_data, test_size=0.2, random_state=42)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = SolarDataset(X_train, y_train, transform)
    test_dataset = SolarDataset(X_test, y_test, transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    model = CNN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 调用 train_model 函数并获取损失和准确率数据
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, train_loader, test_loader,
                                                                               criterion, optimizer, num_epochs=40)

    # 测试模型
    model.eval()
    with torch.no_grad():
        # 使用for循环来控制你想要识别的类别
        for class_idx in range(4):  # 假设有4个类别
            correct = 0
            total = 0
            if class_idx >= 0:
                for images, labels in test_loader:
                    # 仅选择当前循环类别的数据进行测试
                    class_mask = labels == class_idx
                    class_images = images[class_mask]
                    class_labels = labels[class_mask]

                    if len(class_labels) != 0:
                        outputs = model(class_images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += class_labels.size(0)
                        correct += (predicted == class_labels).sum().item()
                if total > 0:
                    print(f'Test Accuracy of the model for class {class_idx}: {100 * correct / total:.2f}%')

    # 绘制准确率曲线
    plt.ylim((0, 1))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('tl\\out\\CNN_accuracy.png', dpi=300)
    plt.show()

    # 绘制损失曲线
    plt.ylim((0, 1))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('CNN Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig('tl\\out\\CNN_loss.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
