import os

import numpy as np
import pandas as pd
import torch

from utils.filter import signal_filter, cwt


def data_processing(dataset_path, is_cwt):
    """
    规整化数据，滤波后转为列表
    :param dataset_path: 数据集路径
    :param is_cwt:是否进行小波变换分析。如果要进行，则不进行之后的所有操作，因为小波变换是为了确认滤波的频率
    :return:[归一化均值为0，方差为1的信号，torch.tensor], [类别，string]一一对应的两个一维列表
    """
    categories = os.listdir(dataset_path)
    signals = []
    labels = []

    if is_cwt:
        for category in categories:
            datas = os.listdir(os.path.join(dataset_path, category))
            cwt(os.path.join(dataset_path, category, datas[0]), os.path.join(os.path.dirname(dataset_path), "小波变换分析图像"))
            cwt(os.path.join(dataset_path, category, datas[-1]), os.path.join(os.path.dirname(dataset_path), "小波变换分析图像"))
        exit(0)

    for category in categories:
        path = os.path.join(dataset_path, category)
        for file_name in os.listdir(path):
            file = os.path.join(dataset_path, category, file_name)
            df = pd.read_csv(file)  # 读取信号
            origin_signal = df.iloc[:, 1].values.astype(np.float32)
            filtered_signal = signal_filter(origin_signal)  # 滤波
            # 归一化
            data = torch.tensor(filtered_signal, dtype=torch.float32)
            normalized_data = (data - torch.mean(data)) / torch.std(data)
            # 添加到列表内
            signals.append(normalized_data)
            labels.append(category)

    return signals, labels
