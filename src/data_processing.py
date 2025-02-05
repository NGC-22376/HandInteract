import os
import pandas as pd
import numpy as np
from utils.filter import signal_filter

def data_processing(dataset_path):
    """
    规整化数据，滤波后转为列表
    :param dataset_path: 数据集路径
    :return:
    """
    categories = os.listdir(dataset_path)
    signals = []
    for category in categories:
        path = os.path.join(dataset_path, category)
        for file_name in os.listdir(path):
            file = os.path.join(dataset_path, category, file_name)
            df = pd.read_csv(file)
            signals.append(signal_filter(df.iloc[:, 1].values.astype(np.float32)))

data_processing(r"C:\Users\30744\Desktop\手互\数据集1.1")