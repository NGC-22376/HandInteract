from utils.csv_visualization import *
from utils.dataset_construction import *

# 建立数据集
data_path = r"C:\Users\30744\Desktop\手互\数据集3"
dataset_path = r"C:\Users\30744\Desktop\手互\数据集"
pdf_path = r"C:\Users\30744\Desktop\手互\手语采样数据集/pdf"
# pdf_construct_dataset(pdf_path, dataset_path)

# 格式化数据集中的数据
format_folders(dataset_path, data_path)

# 对得到的数据做可视化
plot_and_save_csv(dataset_path, 0)