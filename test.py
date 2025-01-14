from utils.csv_visualization import plot_and_save_csv
from utils.dataset_construction import *

csv_path=r"C:\Users\30744\Desktop\手互\手势\测试"
docx_path=r"C:\Users\30744\Desktop\手互\手语采样数据集.docx"
dataset_path=r"C:\Users\30744\Desktop\手互"
# plot_and_save_csv(csv_path,1)
# check_dislocation(docx_path)
docx_construct_dataset(docx_path, dataset_path)