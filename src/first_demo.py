import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 数据集路径
data_dir = r"C:\Users\30744\Desktop\手互\数据集"  # 数据集根目录
gestures = os.listdir(data_dir)  # 获取手势类别名（文件夹名）
gesture_labels = {gesture: idx for idx, gesture in enumerate(gestures)}  # 映射手势到索引

# 初始化存储数据的列表
X, y = [], []

# 遍历每个手势类别
for idx, gesture in enumerate(gestures):
    folder_path = os.path.join(data_dir, gesture)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None)  # 读取CSV文件（假设没有标题）

        # 假设第一列是时间戳，第二列是电压值
        voltage_signal = df.iloc[1:, 1].values.astype(np.float32)  # 强制转换为数值类型（电压值）
        X.append(voltage_signal)
        y.append(gesture_labels[gesture])  # 标签（手势的类别）

# 转换为 NumPy 数组
X = np.array(X, dtype=object)  # X是输入的电压信号
y = np.array(y)  # y是手势标签

# 归一化电压信号
X_min, X_max = [np.min(seq) for seq in X], [np.max(seq) for seq in X]
X = [[((s - X_min[idx]) / (X_max[idx] - X_min[idx])) for s in seq] for idx, seq in enumerate(X)]
X = np.array(X, dtype=object)

# 计算最长序列长度（对输入数据进行填充处理）
max_len = max(len(seq) for seq in X)

# 使用手动填充对数据进行统一长度的填充
X_padded = np.array([np.pad(seq, (0, max_len - len(seq)), mode='constant') for seq in X], dtype=np.float32)

# 转换为PyTorch张量
X_padded = torch.tensor(X_padded, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # PyTorch需要标签为Long类型

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 转换为TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 修改为更大的 batch_size
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 打印钩子函数
def print_shape_hook(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"  Input shape: {input[0].shape}")
    print(f"  Output shape: {output.shape}\n")



# 定义模型
class GestureRecognitionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureRecognitionModel, self).__init__()
        # 卷积层：提取局部特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加channel维度，确保输入形状为 (batch_size, 1, seq_length)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)


        x = torch.relu(self.fc1(x))  # 经过全连接层
        x = self.dropout(x)
        x = self.fc2(x)  # 输出层
        return x


# 模型实例化（不需要考虑GPU设备）
model = GestureRecognitionModel(input_size=1, num_classes=len(gestures))
for layer in model.children():
    layer.register_forward_hook(print_shape_hook)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 调整学习率

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 训练模型
epochs = 50
for epoch in range(epochs):
    print(f"第{epoch + 1}轮评估开始")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        print("1")
        # 无需移到GPU，直接使用CPU
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 更新学习率
    scheduler.step()

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # 无需移到GPU，直接使用CPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
