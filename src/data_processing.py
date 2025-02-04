import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import winsound

def cwt(data_path):
    """
    使用小波变换，分析信号序列的频率构成
    :param data_path: 数据文件路径
    :return:
    """
    # 定义采样频率
    dt = 4e-5
    fs = 1 / dt

    # 获取原始数据
    df = pd.read_csv(data_path, header=None, skiprows=4)
    fmg_signal = df.iloc[:, 1].values.astype(np.float32) * 1000
    t = np.arange(0, len(fmg_signal) * dt, dt)

    # 使用Morlet小波
    wavelet = "morl"
    center_freq = pywt.central_frequency(wavelet)  # Morlet小波中心频率=0.8125

    # 计算尺度范围，得到500个在对数域上均匀分布的尺度值，即最终的结果对低频更敏感
    print("开始小波变换")
    scales = np.logspace(np.log10(center_freq * fs / 100), np.log10(center_freq * fs / 0.1), num=500)
    coefficients, frequencies = pywt.cwt(fmg_signal, scales, wavelet, sampling_period=1 / fs)

    # 提取不同频段的子集
    mask_to1 = (frequencies >= 0.1) & (frequencies <= 1)
    mask_to10 = (frequencies > 1) & (frequencies <= 10)
    mask_to50 = (frequencies > 10) & (frequencies <= 50)
    mask_to100 = (frequencies > 50) & (frequencies <= 100)

    # 绘图展示
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(231)
    plt.plot(t, fmg_signal)
    plt.xlabel("Time/s")
    plt.ylabel("Voltage/mV")
    plt.title("Raw Signal")

    # 小波时频图
    def draw_after_transform(pos, coefficients, frequencies, levels, begin, end):
        plt.subplot(pos)
        plt.contourf(t, frequencies, np.abs(coefficients), levels=levels, cmap="jet")
        plt.title(f"After Transform:{begin}Hz-{end}Hz")
        plt.xlabel("Time/s")
        plt.ylabel("Frequency/Hz")
        plt.ylim(begin, end)
        plt.colorbar(label='Coefficient Magnitude')

    draw_after_transform(233, coefficients[mask_to1], frequencies[mask_to1], 20, 0.1, 1)
    draw_after_transform(234, coefficients[mask_to10], frequencies[mask_to10], 20, 1, 10)
    draw_after_transform(235, coefficients[mask_to50], frequencies[mask_to50], 80, 10, 50)
    draw_after_transform(236, coefficients[mask_to100], frequencies[mask_to100], 100, 50, 100)

    plt.tight_layout()
    plt.show()

    winsound.Beep(1000, 1000)

cwt(r"C:\Users\30744\Desktop\手互\数据集3\大拇指\大拇指（1.5）.csv")