import os

import torch

from utils.extract_feature import get_feature_window
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
        os.makedirs(os.path.join(os.path.dirname(dataset_path), "小波变换分析图像"))
        for category in categories:
            datas = os.listdir(os.path.join(dataset_path, category))
            cwt(os.path.join(dataset_path, category, datas[0]),
                os.path.join(os.path.dirname(dataset_path), "小波变换分析图像"))
            cwt(os.path.join(dataset_path, category, datas[-1]),
                os.path.join(os.path.dirname(dataset_path), "小波变换分析图像"))
        exit(0)

    for idx, category in enumerate(categories):
        path = os.path.join(dataset_path, category)
        signals.append([])
        labels.append([])
        for file_name in os.listdir(path):
            file = os.path.join(dataset_path, category, file_name)
            filtered_signal = signal_filter(file)  # 滤波
            # 归一化
            data = torch.tensor(filtered_signal.copy(), dtype=torch.float32)
            normalized_data = (data - torch.mean(data)) / torch.std(data)
            # 滑动窗口+特征提取
            feature_windows, window_num = get_feature_window(normalized_data, window_size=32)
            # 添加到列表内
            signals[-1].extend(feature_windows)
            labels[-1].extend([idx] * window_num)
    return signals, labels

