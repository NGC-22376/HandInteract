"""
窗口化数据和提取特征
"""
import numpy as np
from scipy.stats import skew, kurtosis

def get_feature_window(signal, window_size):
    """
    窗口化数据，并对每一个窗口提取特征
    :param signal: 经过滤波和归一化后的信号
    :param window_size: 每一个窗口的长度
    :return: [[], [], []...[]]形状的窗口列表，以及窗口的总个数
    """
    # 滑动窗口，丢弃最后一个不足长度的窗口
    window_num = len(signal) // window_size
    windows = [np.array(signal[i:i + window_size]) for i in range(0, window_num)]
    features = []
    for window in windows:
        features.append([])
        for i in window:
            features[-1].append([i] * 10)
    #     features.append([])
    #     # 计算 10 个特征
    #     features[-1].append(np.mean(window))  # 均值
    #     features[-1].append(np.var(window))  # 方差
    #     features[-1].append(np.std(window))  # 标准差
    #     features[-1].append(np.max(window))  # 最大值
    #     features[-1].append(np.min(window))  # 最小值
    #     features[-1].append(np.sqrt(np.mean(window ** 2)))  # 均方根 (RMS)
    #     features[-1].append(skew(window))  # 偏度
    #     features[-1].append(kurtosis(window))  # 峭度
    #     features[-1].append(np.max(window) - np.min(window))  # 峰峰值
    #     features[-1].append(np.std(window) / np.mean(window) if np.mean(window) != 0 else 0)  # 变异系数
    return features, window_num
