import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
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
X_min, X_max = [np.min(seq) for seq in X],  [np.max(seq) for seq in X]
X = [[((s - X_min[idx]) / (X_max[idx] - X_min[idx])) for s in seq] for idx, seq in enumerate(X)]
X = np.array(X, dtype=object)
# 计算最长序列长度（对输入数据进行填充处理）
max_len = max(len(seq) for seq in X)

# 使用 pad_sequences 对数据进行填充，确保输入数据的统一长度
X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# 独热编码标签（转换为0-1编码形式，适用于分类问题）
y = to_categorical(y, num_classes=len(gestures))

# 构建模型
model = Sequential([
    # 卷积层：提取电压信号的局部特征
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(max_len, 1)),
    MaxPooling1D(pool_size=2),

    # 第二层卷积层：提取更复杂的特征
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    # LSTM 层：捕捉时序特征，分析信号的时间演变
    LSTM(100, return_sequences=False),

    # 全连接层：将提取的特征进行映射并用于分类
    Dense(64, activation='relu'),
    Dropout(0.5),  # Dropout层防止过拟合

    # 输出层：类别数等于手势类别数，使用Softmax激活函数输出概率分布
    Dense(len(gestures), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 评估模型
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
