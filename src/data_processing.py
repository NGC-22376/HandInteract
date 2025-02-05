import os
import pandas as pd
import numpy as np
import torch
from utils.filter import signal_filter

def data_processing(dataset_path):
    """
    规整化数据，滤波后转为列表
    :param dataset_path: 数据集路径
    :return:[归一化均值为0，方差为1的信号，torch.tensor], [类别，string]一一对应的两个一维列表
    """
    categories = os.listdir(dataset_path)
    signals = []
    labels = []
    for category in categories:
        path = os.path.join(dataset_path, category)
        for file_name in os.listdir(path):
            file = os.path.join(dataset_path, category, file_name)
            df = pd.read_csv(file) # 读取信号
            filtered_signal = signal_filter(df.iloc[:, 1].values.astype(np.float32)) # 滤波
            # 归一化
            data = torch.tensor(filtered_signal, dtype=torch.float32)
            normalized_data = (data - torch.mean(data)) / torch.std(data)
            # 添加到列表内
            signals.append(normalized_data)
            labels.append(category)

    return signals, labels