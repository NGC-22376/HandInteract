import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.signal import butter, filtfilt

from utils.visualization import draw_signal, print_msg


def signal_filter(data_path):
    """
    中值滤波->高通滤波->低通滤波->降采样
    :param data_path:存储原始信号的文件路径
    :return:滤波后的信号
    """
    df = pd.read_csv(data_path)  # 读取信号
    origin_signal = df.iloc[:, 1].values.astype(np.float32)
    fs = 1 / 4e-5
    fmax = 110
    nyquist = fs / 2
    cutoff_high = 2.5 * fmax / 2  # 低频滤波截止频率，要求大于最高频的奈奎斯特频率
    cutoff_low = 1 / nyquist  # 标准化高通滤波截止频率

    # 可视化
    # plt.figure(figsize=(12, 8))
    # draw_signal(origin_signal, 2, 2, 1, "Raw Signal")

    # 中值滤波-去除尖峰
    signal.medfilt(origin_signal, kernel_size=5)
    # draw_signal(origin_signal, 2, 2, 2, "After 中值滤波")

    # 高通滤波-去除运动伪影
    b, a = butter(4, cutoff_low, btype='high')
    filtfilt(b, a, origin_signal)
    # draw_signal(origin_signal, 2, 2, 3, "After 高通滤波")

    # 低通滤波+降采样
    filtered = signal.decimate(origin_signal, int(fs // cutoff_high), ftype="iir")
    # draw_signal(filtered, 2, 2, 4, "After 低通滤波和降采样", dt=1 / cutoff_high)

    # 展示图象
    # plt.tight_layout()
    # plt.show()
    return filtered


def cwt(data_path, img_dir):
    """
    使用小波变换，分析信号序列的频率构成
    :param data_path: 数据文件路径
    :param img_dir: 小波变换图象的存储路径文件夹
    :return:
    """
    # 定义采样频率
    dt = 4e-5
    fs = 1 / dt

    # 获取原始数据
    type_name = os.path.basename(os.path.dirname(data_path))
    df = pd.read_csv(data_path, header=None, skiprows=4)
    fmg_signal = df.iloc[:, 1].values.astype(np.float32)
    t = np.arange(0, len(fmg_signal) * dt, dt)

    # 使用Morlet小波
    wavelet = "morl"
    center_freq = pywt.central_frequency(wavelet)  # Morlet小波中心频率=0.8125

    # 计算尺度范围，得到50个在对数域上均匀分布的尺度值，即最终的结果对低频更敏感
    print_msg(f"开始小波变换：{type_name}")
    scales = np.logspace(np.log10(center_freq * fs / 100), np.log10(center_freq * fs / 0.1), num=50)
    coefficients, frequencies = pywt.cwt(fmg_signal, scales, wavelet, sampling_period=1 / fs)

    # 提取不同频段的子集。布尔掩码列表，按位判断frequencies中的每个值满足条件与否，再按位将两个掩码表做&运算
    mask_to1 = (frequencies >= 0.1) & (frequencies <= 1)
    mask_to10 = (frequencies > 1) & (frequencies <= 10)
    mask_to50 = (frequencies > 10) & (frequencies <= 50)
    mask_to100 = (frequencies > 50) & (frequencies <= 100)
    mask_over100 = frequencies > 100

    # 绘图展示
    plt.figure(figsize=(15, 10))

    # 原始图像
    draw_signal(fmg_signal, 3, 2, 1, "Raw Signal " + type_name)

    # 小波时频图
    def draw_after_transform(pos, coefficients, frequencies, levels, begin, end):
        plt.subplot(pos)
        plt.contourf(t, frequencies, np.abs(coefficients), levels=levels, cmap="jet")
        plt.title(f"After Transform:{begin}Hz-{end}Hz")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        plt.ylim(begin, end)
        plt.colorbar(label='Coefficient Magnitude')

    # 保证打印非空频率
    if np.any(mask_to1):
        draw_after_transform(323, coefficients[mask_to1], frequencies[mask_to1], 20, 0.1, 1)
    if np.any(mask_to10):
        draw_after_transform(324, coefficients[mask_to10], frequencies[mask_to10], 20, 1, 10)
    if np.any(mask_to50):
        draw_after_transform(325, coefficients[mask_to50], frequencies[mask_to50], 80, 10, 50)
    if np.any(mask_to100):
        draw_after_transform(326, coefficients[mask_to100], frequencies[mask_to100], 100, 50, 100)
    if np.any(mask_over100):
        print(f"最高频率：{frequencies[mask_over100]}")

    plt.tight_layout()

    # 保存并显示图象
    plt.savefig(os.path.join(img_dir, type_name + os.path.basename(data_path) + ".png"))
    plt.show()
