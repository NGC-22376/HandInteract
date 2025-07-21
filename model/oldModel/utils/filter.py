import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, decimate, firwin, lfilter

# 生成模拟数据（假设4通道，每个通道10秒数据）
fs_original = 25000  # 原始采样率
dataset_plot_path = r"E:\UESTC\手互\output"
dataset_path = r"E:\UESTC\手互\output"
# 参数设置
target_fs = 100  # 目标采样率（降采样后）
nyquist_original = fs_original / 2
nyquist_new = target_fs / 2
# 设计低通滤波器（示例参数）
lp_cutoff = 10.0
order_lowpass = 4


def factorize(n, max_stage):
    factors = []
    while n > 1:
        found = False
        # 从最大值向小寻找可分解因子
        for f in range(min(max_stage, n), 1, -1):
            if n % f == 0:
                factors.append(f)
                n = n // f
                found = True
                break
        if not found:  # 无法分解时取剩余值
            factors.append(n)
            break
    return factors
# 步骤1：抗混叠滤波 + 降采样
# -----------------------------------------------
def downsample(data, original_fs, target_fs):
    """降采样函数"""
    total_factor = int(original_fs // target_fs)
    print(f'降采样倍数: {total_factor}')

    # 分解因子（确保乘积等于总降采样倍数）
    factors = factorize(total_factor, max_stage=10)
    print(f"总降采样倍数: {total_factor} → 分阶段倍数: {factors}")

    # ========== 分阶段降采样 ==========
    downsampled = data.copy()
    for idx, f in enumerate(factors, 1):
        required_len = 30  # decimate最小需求长度
        if len(downsampled) < required_len:
            raise ValueError(
                f"第{idx}阶段降采样前数据长度不足({len(downsampled)} < {required_len})"
                f"，请检查数据长度或调整分阶段策略"
            )
        safety_margin=5
        # 动态调整滤波器阶数
        filter_order = min(8, int(len(downsampled) / f - safety_margin))
        downsampled = decimate(
            downsampled,
            f,
            n=filter_order,  # 自动调整阶数
            zero_phase=True
        )

    return downsampled


# 步骤2：带通滤波（0.1-10Hz）
# -----------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=6):
    """生成带通滤波器系数"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filter(data, filter_func, fs):
    """应用零相位滤波器"""
    b, a = filter_func
    return filtfilt(b, a, data)


# 步骤3：可选的低通滤波（如果只需要<10Hz）
# -----------------------------------------------
def butter_lowpass(cutoff, fs, order=5):
    """生成低通滤波器系数"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a


# 可视化结果
# -----------------------------------------------
def plot_comparison(original, filtered, fs_orig, fs_new, show, plot_path):
    """绘制原始和滤波后信号对比"""
    t_orig = np.arange(original.shape[0]) / fs_orig
    t_new = np.arange(filtered.shape[0]) / fs_new

    plt.figure(figsize=(12, 8))

    # 时域对比
    plt.subplot(2, 1, 1)
    plt.plot(t_orig, original[:], alpha=0.5, label='Original')
    plt.plot(t_new, filtered[:], label='Filtered')
    plt.xlim(0, 3.5)  # 显示前2秒
    plt.xlabel('Time (s)')
    plt.legend()

    # 频域对比
    plt.subplot(2, 1, 2)
    n_fft = 4096
    f_orig = np.fft.rfftfreq(n_fft, 1 / fs_orig)
    f_new = np.fft.rfftfreq(n_fft, 1 / fs_new)
    plt.plot(f_orig, np.abs(np.fft.rfft(original[ :n_fft], n_fft)),
             alpha=0.5, label='Original', color='blue', linestyle='--')
    # 滤波后信号频谱
    plt.plot(f_new, np.abs(np.fft.rfft(filtered[:n_fft], n_fft)),
             label='Filtered', color='red', linewidth=1.5)
    plt.xlim(0, 50)  # 显示0-50Hz范围
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)

    if show:
        plt.show()

def signal_filter(origin_signal):
    #输入（信号长度，通道数）的原始信号，输出（信号长度,通道数）的过滤且降采样信号
    data_filtered=[]
    for channel in range(n_channels):
        raw_data_relate = origin_signal[:,channel] - np.mean(origin_signal[:,channel])
        # 设计抗混叠滤波器（Butterworth低通滤波器，截止频率为nyquist_new）
        num_taps = 101
        fir_coeff = firwin(num_taps, nyquist_new, fs=fs_original, pass_zero="lowpass")
        data_pre = filtfilt(fir_coeff, 1.0, raw_data_relate)
        data_downsampled = np.array(downsample(data_pre, fs_original, target_fs))

        b_low, a_low = butter_lowpass(lp_cutoff, target_fs, order_lowpass)
        data_filtered_channel = np.array(apply_filter(data_downsampled, (b_low, a_low), target_fs))
        # raw_data_repeat4为（36329,4）data_filtered
        data_filtered.append(data_filtered_channel.reshape(-1, 1))
    return np.hstack(data_filtered)


categories = os.listdir(dataset_path)
n_channels = 1  # 考虑输入为4通道信号，即origin_signal形状为（信号长度，4）

for idx, category in enumerate(categories):
    path = os.path.join(dataset_path, category)
    for file_name in os.listdir(path):
        file = os.path.join(dataset_path, category, file_name)
        df = pd.read_csv(file)  # 读取信号
        origin_signal = df.iloc[:, 1:n_channels + 1].values.astype(np.float32)  # (86329,)
        filtered = signal_filter(origin_signal)
        for channel in range(n_channels):
            plot_path = os.path.join(dataset_plot_path + category + os.path.basename(file_name) + f"_{channel}通道.png")
            plot_comparison(origin_signal[:,channel], filtered[:,channel], fs_original, target_fs,0,plot_path)

