"""
数据预处理：
小波变换分析信号中大概的成分
低通滤波滤除传感器等带来的高频噪声干扰
中值滤波滤除峰值噪声，并保留信号的上升和下降趋势
"""
import numpy
from scipy import signal
import pywt

def cwt(data_path):
    """
    使用小波变换，分析信号序列的频率构成
    :param data_path:
    :return:
    """


