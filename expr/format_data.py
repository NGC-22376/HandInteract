from utils.dataset_construction import format_folders
from utils.visualization import plot_and_save_csv

dataset_path =  # 数据集路径
data_path =  # 原始数据路径

format_folders(dataset_path, data_path)
plot_and_save_csv(dataset_path, is_show=1)
