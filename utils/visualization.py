"""
将采样得到的csv文件构成的数据集进行可视化，输出为图片保存于特定的路径下
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置默认字体为黑体（SimHei 适用于 Windows，AppleGothic 适用于 Mac）
rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # SimHei 显示中文，Arial 作为备用
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题



def plot_and_save_csv(dir_path, is_show):
    """
    适用数据集格式：取样轮次/手势名/该手势的多个实验数据。如：第一次试采/谢谢/样本1.csv。
    :param dir_path: 手势名的上一级完整目录，如：C:/.../上一级目录/第一次试采。
    :param is_show:是否在边栏展示绘制的图像。
    :return: 无返回值。该函数执行完后，数据集的格式为：上一级目录/取样轮次/手势名/该手势的多个实验数据，上一级目录/取样轮次+可视化文件/手势名/该手势的多个csv可视化文件。
    """
    dir_names = os.listdir(dir_path)
    for dir_name in dir_names:
        for index, file_name in enumerate(os.listdir(os.path.join(dir_path, dir_name))):
            file = os.path.join(dir_path, dir_name, file_name)  # 设置csv文件路径
            df = pd.read_csv(file, dtype={0: 'float', 1: 'float'}, skiprows=4,
                             names=['时间', '电压强度'])  # 通过pandas读取csv文件内容，跳过非数据行

            # 获取横纵坐标最值
            y_max = -10.
            y_min = 10.
            for y in df.values[:, 1]:
                y_max = max(y_max, y)
                y_min = min(y_min, y)

            print(f"{file_name}:第{index}组数据可视化完成")

            # 绘图
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 防止中文乱码
            plt.plot(df.values[:, 0], df.values[:, 1])  # 设置横纵轴数据
            plt.xlabel('时间/s')  # 设置横坐标名称
            plt.ylabel('信号强度/V')  # 设置纵坐标名称
            plt.title(f'动作信号强度：{dir_name}-{file_name}')  # 设置图表标题
            plt.axis([df.values[0:1][:, 0], df.values[-1:][:, 0], y_min, y_max])  # 设置xy轴范围

            # 保存图片
            parent_path = os.path.join(os.path.dirname(dir_path), dir_path + "-Vis", dir_name)
            name = f'{index + 1}' + '.png'
            os.makedirs(parent_path, exist_ok=True)
            save_path = os.path.join(parent_path, name)
            plt.savefig(save_path)

            # 展示
            if is_show:
                plt.show()
            else:
                plt.close()


def draw_signal(signal, line, row, idx, name, dt=4e-5):
    """
    绘制信号图像（包括子图）
    :param signal: 信号
    :param name: 图像名
    :param pos: 处于网格图的位置
    :param dt: 信号的时间间隔
    :return:
    """
    t = np.arange(0, len(signal) * dt, dt)
    plt.subplot(line, row, idx)
    plt.plot(t, signal)
    plt.xlabel("Time/s")
    plt.ylabel("Voltage/V")
    plt.title(name)


def print_msg(msg):
    t = datetime.now()
    print(f"{t.strftime('%Y-%m-%d %H:%M:%S')}-{msg}")


