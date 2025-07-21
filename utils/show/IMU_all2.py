#30条数据的时候79%，10条的时候78.30%!
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pywt
from scipy.special import softmax
# 残差没区别

from collections import defaultdict

from collections import deque
from sklearn.svm import SVC
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
def load_data(data_dir):
    X = []
    y = []
    original_lengths = []
    label_map = {folder: idx for idx, folder in enumerate(os.listdir(data_dir))}

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label = label_map[folder]
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                    data = np.hstack((data[:, 1:10], data[:, 16:20]))
                    original_length = len(data)
                    original_lengths.append(original_length)
                    X.append(data)
                    y.append(label)

    max_length = max(original_lengths)
    X_padded = [np.pad(sample, ((0, max_length - len(sample)), (0, 0)), mode='constant') for sample in X]
    X_padded = np.array(X_padded, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"数据加载完成，X 共有 {len(X_padded)} 个样本，最大长度 {max_length}，每个样本形状 {X_padded.shape[1:]}")
    print(f"标签映射: {label_map}")

    return X_padded, y, original_lengths

def create_sliding_windows(data, labels, original_lengths, window_size=20, step_size=5):
    X, y = [], []
    for i in range(len(data)):
        single_data = data[i]
        original_length = original_lengths[i]
        # 仅对原始数据部分生成窗口
        for j in range(0, original_length - window_size, step_size):
            X.append(single_data[j:j + window_size])
            y.append(labels[i])
    return np.array(X), np.array(y)

def wavelet_transform(x, wavelet='db4', level=3):
    coeffs = pywt.wavedec(x, wavelet, level=level, axis=1)
    return np.concatenate(coeffs, axis=1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim):
        super(Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, seq_len * hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.seq_len, self.hidden_dim)
        x = self.decoder(x.view(-1, self.seq_len * self.hidden_dim))
        return x

class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn_weights * x, dim=1)

class DeepConvBiLSTM(nn.Module):
    def __init__(self, input_size=32, conv_filters=64, lstm_hidden=128, num_classes=10, num_lstm_layers=2, dropout=0.5):
        super(DeepConvBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, kernel_size=5, stride=1, padding=2)
        self.res_block1 = ResidualBlock(conv_filters, conv_filters)
        self.res_block2 = ResidualBlock(conv_filters, conv_filters * 2)
        self.bilstm = nn.LSTM(conv_filters * 2, lstm_hidden, num_layers=num_lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(lstm_hidden * 2)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len) -> (batch, seq_len, features)
        x, _ = self.bilstm(x)
        x = self.attention(x)
        x = self.fc(self.dropout(x))
        return x

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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = r'D:\WeChatfiles\WeChat Files\wxid_u266ve7902wp22\FileStorage\File\2025-07\数据 - 副本'
    X_original, y_original, original_lengths = load_data(data_dir)
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X_original, y_original, test_size=0.2, stratify=y_original, random_state=42
    )
    train_indices, test_indices = train_test_split(
        np.arange(len(X_original)), test_size=0.2, stratify=y_original, random_state=42
    )
    print(y_test_original)
    original_lengths_train = [original_lengths[i] for i in range(len(X_original)) if i in train_indices]
    original_lengths_test = [original_lengths[i] for i in range(len(X_original)) if i in test_indices]

    X_train, y_train = create_sliding_windows(X_train_original, y_train_original, original_lengths_train)
    X_test, y_test = create_sliding_windows(X_test_original, y_test_original, original_lengths_test)

    X_train_wavelet = np.array([wavelet_transform(x) for x in X_train])
    X_test_wavelet = np.array([wavelet_transform(x) for x in X_test])

    print(f"X_train_wavelet shape before reshape: {X_train_wavelet.shape}")
    X_train_wavelet = X_train_wavelet.reshape(X_train_wavelet.shape[0], -1)  # 展平为 (batch_size, seq_len * input_dim)
    print(f"X_train_wavelet shape after reshape: {X_train_wavelet.shape}")

    print(f"X_test_wavelet shape before reshape: {X_test_wavelet.shape}")
    X_test_wavelet = X_test_wavelet.reshape(X_test_wavelet.shape[0], -1)  # 展平为 (batch_size, seq_len * input_dim)
    print(f"X_test_wavelet shape after reshape: {X_test_wavelet.shape}")

    scaler = StandardScaler()
    X_train_wavelet = scaler.fit_transform(X_train_wavelet)
    X_test_wavelet = scaler.transform(X_test_wavelet)
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)

    input_dim = X_train_wavelet.shape[1]
    seq_len = 20  # 滑动窗口大小
    hidden_dim = 32
    autoencoder = Autoencoder(input_dim, seq_len, hidden_dim)
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion_ae = nn.MSELoss()
    autoencoder.load_state_dict(torch.load("D:\\WeChatfiles\\WeChat Files\\wxid_u266ve7902wp22\\FileStorage\\File\\2025-06\\output\\imu_encoder.pth"))

    for epoch in range(100):
        autoencoder.train()
        optimizer_ae.zero_grad()
        outputs = autoencoder(torch.tensor(X_train_wavelet, dtype=torch.float32))
        loss = criterion_ae(outputs, torch.tensor(X_train_wavelet, dtype=torch.float32))
        loss.backward()
        optimizer_ae.step()

        autoencoder.eval()
        with torch.no_grad():
            val_outputs = autoencoder(torch.tensor(X_test_wavelet, dtype=torch.float32))
            val_loss = criterion_ae(val_outputs, torch.tensor(X_test_wavelet, dtype=torch.float32))
        print(f'Epoch {epoch + 1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')
    torch.save(autoencoder.state_dict(),  f"D:\\WeChatfiles\\WeChat Files\\wxid_u266ve7902wp22\\FileStorage\\File\\2025-06"
                                   f"\\output\\imu_encoder2.pth")

    # 降维
    with torch.no_grad():
        X_train_reduced = autoencoder.encoder(torch.tensor(X_train_wavelet, dtype=torch.float32)).numpy()
        X_test_reduced = autoencoder.encoder(torch.tensor(X_test_wavelet, dtype=torch.float32)).numpy()

    window_size = 20
    X_train_reduced = X_train_reduced.reshape(-1, window_size, hidden_dim)
    X_test_reduced = X_test_reduced.reshape(-1, window_size, hidden_dim)

    train_dataset = IMUDataset(X_train_reduced, y_train)
    test_dataset = IMUDataset(X_test_reduced, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DeepConvBiLSTM(num_classes=len(np.unique(y_train))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_acc=0
    #model.load_state_dict(torch.load(f"D:\\WeChatfiles\\WeChat Files\\wxid_u266ve7902wp22\\FileStorage\\File\\2025-06"
    #                               f"\\output\\imu_model.pth"))

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
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"D:\\WeChatfiles\\WeChat Files\\wxid_u266ve7902wp22\\FileStorage\\File\\2025-06"
                                   f"\\output\\imu_model2.pth")
            print(f"Best model updated and saved at epoch {epoch + 1} with acc {test_acc:.2f}%!")

def eval(csv_dir):# 测试集评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 20

    hidden_dim = 32
    batch_size = 32

    def load_stream_csv(csv_dir):
        raw_data = np.genfromtxt(csv_dir, delimiter=',', skip_header=1)
        # 对齐训练时选取的特征列
        selected_data = np.hstack((raw_data[:, 1:10], raw_data[:, 16:20]))  # (n_samples, 13)
        dummy_labels = [0]  # dummy 占位
        lengths = [len(selected_data)]
        return selected_data,dummy_labels,lengths

    X_original, y_dummy, lengths = load_stream_csv(csv_dir)

    # 2. 滑动窗口 + 小波变换
    X_slided, _ = create_sliding_windows(X_original, y_dummy, lengths)
    X_wavelet = np.array([wavelet_transform(x) for x in X_slided])
    X_wavelet = X_wavelet.reshape(X_wavelet.shape[0], -1)

    # 3. 加载 scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load("scaler_mean.npy")
    scaler.scale_ = np.load("scaler_scale.npy")
    X_scaled = scaler.transform(X_wavelet)

    # 4. 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = Autoencoder(input_dim=X_scaled.shape[1], seq_len=seq_len, hidden_dim=hidden_dim)
    autoencoder.load_state_dict(torch.load("imu_encoder2.pth", map_location=device))
    autoencoder.eval().to(device)

    model = DeepConvBiLSTM(num_classes=7).to(device)
    model.load_state_dict(torch.load("imu_model2.pth", map_location=device))
    model.eval()

    # 5. 降维 + reshape 成分类输入格式
    with torch.no_grad():
        encoded = autoencoder.encoder(torch.tensor(X_scaled, dtype=torch.float32).to(device))
        X_final = encoded.reshape(-1, seq_len, hidden_dim)

    # 6. 批量预测
    dataset = IMUDataset(X_final, np.zeros(X_final.shape[0]))  # 标签是 dummy
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # 输出结果
    print("预测标签序列：", all_preds)
    return all_preds