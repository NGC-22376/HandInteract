from utils.visualization import plot_and_save_csv
from utils.dataset_construction import format_folders
from data_processing import data_processing

# 建立数据集
data_path = r"C:\Users\30744\Desktop\手互\数据集2"
dataset_path = r"C:\Users\30744\Desktop\手互\数据集2.1"
pdf_path = r"C:\Users\30744\Desktop\手互\手语采样数据集.pdf"
# pdf_construct_dataset(pdf_path, dataset_path)

# 格式化数据集中的数据
# format_folders(dataset_path, data_path)

# 对得到的数据做可视化
# plot_and_save_csv(dataset_path, 0)

# 数据预处理
X, y = data_processing(dataset_path, 1)