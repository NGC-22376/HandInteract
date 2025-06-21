import os

import numpy as np

from oldModel.utils.filter import signal_filter
from utils.visualization import plot_and_save_csv

dataset_path = r"C:\Users\30744\Desktop\手互\Dataset-2"  # 数据集路径
data_path = r"E:\李佳乐"  # 原始数据路径

# format_folders(dataset_path, data_path)
# plot_and_save_csv(data_path, is_show=1)
for i in os.listdir(data_path):
    dir_path = os.path.join(data_path, i)
    for j in os.listdir(dir_path):
        file_path = os.path.join(dir_path, j)
        signal_filter(file_path)

