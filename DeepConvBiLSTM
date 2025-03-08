import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: 加载数据集
def load_your_data(data_dir):
    X = []
    y = []
    original_lengths = []  # 新增：记录每个样本的原始长度
    label_map = {folder: idx for idx, folder in enumerate(os.listdir(data_dir))}

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label = label_map[folder]
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)[:, 1:22]
                    original_length = len(data)  # 记录原始长度
                    original_lengths.append(original_length)
                    X.append(data)
                    y.append(label)

    max_length = max(original_lengths)
    X_padded = [np.pad(sample, ((0, max_length - len(sample)), (0, 0)), mode='constant') for sample in X]
    X_padded = np.array(X_padded, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"数据加载完成，X 共有 {len(X_padded)} 个样本，最大长度 {max_length}，每个样本形状 {X_padded.shape[1:]}")
    print(f"标签映射: {label_map}")  # 输出标签对应关系

    return X_padded, y, original_lengths



# Step 2: 滑动窗口处理
def create_sliding_windows(data, labels, original_lengths, window_size=20, step_size=5):
    X, y = [], []
    for i in range(len(data)):
        single_data = data[i]
        original_length = original_lengths[i]  # 获取该样本的原始长度
        # 仅对原始数据部分生成窗口（忽略填充的0）
        for j in range(0, original_length - window_size, step_size):
            X.append(single_data[j:j + window_size])
            y.append(labels[i])
    return np.array(X), np.array(y)


# Step 3: 构建 PyTorch 数据集
class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Step 4: 定义 DeepConvBiLSTM 模型
class DeepConvBiLSTM(nn.Module):
    def __init__(self, input_size=21, conv_filters=64, lstm_hidden=128, num_classes=10, num_lstm_layers=2, dropout=0.5):
        super(DeepConvBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, stride=1, padding=2)
        self.bn = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden, num_layers=num_lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) -> (batch, seq_len, features)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = self.fc(self.dropout(x))
        return x


# Step 5: 训练模型
def train_model(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f'Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    return train_loss, train_acc


# Step 6: 测试模型
def test_model(test_loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    return test_loss, test_acc


# Step 7: 主程序

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'D:\大二下\互联网加\位置传感器\\data'
    X_original, y_original, original_lengths = load_your_data(data_dir)

    # 划分原始样本
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X_original, y_original, test_size=0.2, stratify=y_original, random_state=42
    )
    train_indices, test_indices = train_test_split(
        np.arange(len(X_original)), test_size=0.2, stratify=y_original, random_state=42
    )

    # 生成窗口时使用原始长度（需从load_your_data返回original_lengths）
    # 注意：此处需要传递训练集和测试集对应的original_lengths
    # 假设 original_lengths 是完整数据集的长度列表，需同步划分
    original_lengths_train = [original_lengths[i] for i in range(len(X_original)) if i in train_indices]
    original_lengths_test = [original_lengths[i] for i in range(len(X_original)) if i in test_indices]

    X_train, y_train = create_sliding_windows(X_train_original, y_train_original, original_lengths_train)
    X_test, y_test = create_sliding_windows(X_test_original, y_test_original, original_lengths_test)

    # 归一化处理（关键修改点：先归一化再创建数据集）
    scaler = StandardScaler()
    # 训练集归一化
    num_train_samples, window_size, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    scaler.fit(X_train_reshaped)
    X_train_normalized = scaler.transform(X_train_reshaped).reshape(num_train_samples, window_size, num_features)

    # 测试集归一化
    num_test_samples, _, _ = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_features)
    X_test_normalized = scaler.transform(X_test_reshaped).reshape(num_test_samples, window_size, num_features)

    # 创建归一化后的数据集
    train_dataset = IMUDataset(X_train_normalized, y_train)  # 使用归一化后的数据
    test_dataset = IMUDataset(X_test_normalized, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DeepConvBiLSTM(num_classes=len(np.unique(y_train))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = running_loss / len(train_loader), 100 * correct / total
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
        scheduler.step(train_loss)

        # 测试集评估
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_loss, test_acc = test_loss / len(test_loader), 100 * correct / total
        print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    torch.save(model.state_dict(), f"D:\\大二下\\互联网加\\位置传感器\\log\\params_epoch_{epoch + 1}.pth")
    print(f"Model saved for epoch {epoch + 1}!")


if __name__ == "__main__":
    main()
