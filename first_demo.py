import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 假设数据集结构为：
# your_dataset_directory/
# ├── 手势名/
# │   ├── sample1.xlsx
# │   ├── sample2.xlsx
# │   ├──
# │   ├── sample20.xlsx


# 1.读取数据
# 使用 pandas 读取 Excel 文件，并用 os 遍历所有文件。
data_dir = r"C:\Users\30744\Desktop\手互\数据集"  # 数据集根目录
gestures = os.listdir(data_dir)  # 获取手势类别名（文件夹名）“gesture1”
gesture_labels = {gesture: idx for idx, gesture in enumerate(gestures)}  # 映射手势到索引

# 初始化
X, y = [], []

# 遍历手势数据
for gesture in gestures:
    folder_path = os.path.join(data_dir, gesture)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None)  # 读取csv文件
        voltage_signal = df.iloc[:, 1].values  # 取第二列（电压）
        X.append(voltage_signal)
        y.append(gesture_labels[gesture])

# 转换为 NumPy 数组，确保每个序列是对象类型的数组
X = np.narray(X, dtype=object)  # X是输入的电压信号
y = np.narray(y)  # y是对应的手势标签

# 归一化
X_min, X_max = np.min([np.min(seq) for seq in X]), np.max([np.max(seq) for seq in X])
X = [(seq - X_min) / (X_max - X_min) for seq in X]

# 计算最长序列长度
max_len = max(len(seq) for seq in X)

# 使用 pad_sequences 对 X 进行填充
X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')

# 独热编码标签：将标签转换为 0-1 编码，方便模型学习
y = to_categorical(y, num_classes=len(gestures))

# 2.构建模型:使用 CNN + LSTM 结合 进行特征提取和时序学习
model = Sequential([
    # 卷积层:提取电信号的局部特征，电压波动模式
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(max_len, 1)),
    # 池化层:进一步减少数据维度，保留最显著的特征
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    # LSTM 层:时序学习，提取时序特征，适用于电压信号这样的时序数据。
    # 使用 LSTM 层捕捉信号的 时序依赖，理解手势动作的时间演变。
    LSTM(100, return_sequences=False),

    # 全连接层:输出层，将 LSTM 层输出的特征映射到手势类别上
    # 全连接层用于信息整合，把前面提取的特征结合在一起，输出一个具有64个神经元的向量，表示特征空间的一个新的表示
    Dense(64, activation='relu'),

    # Dropout:防止过拟合，随机丢弃一些神经元，减少模型对训练数据的依赖
    Dropout(0.5),

    # 输出层:输出神经元的数量等于手势的类别数，通过 Softmax 输出层将特征映射到每个手势类别，输出每个手势的概率分布，从中选择概率最高的手势作为预测结果
    Dense(len(gestures), activation='softmax')  # 输出类别数
])

# 编译模型: 优化器、损失函数、评估指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3.模型训练与评估
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 评估模型
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
