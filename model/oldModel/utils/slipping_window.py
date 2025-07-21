import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

dataset_path = r"D:\download\feishu_download\dataset"
n_channels=1
classes = 8
def get_feature_window(subject_num):
    categories = os.listdir(dataset_path)
    for traget_dataset in categories:
        match = re.match(r'^(\d+)_', traget_dataset)
        if match:
            num = int(match.group(1))  # 提取第一个捕获组的数字并转为整数
            if num == subject_num:
                break

    data_all=[[] for _ in range(classes)]
    label_all=[[] for _ in range(classes)]
    data_split_point = []
    datanum_all=[]

    path1 = os.path.join(dataset_path, traget_dataset)#'D:\\download\\feishu_download\\dataset\\0_dataset_佳乐'
    print(path1)
    for category in os.listdir(path1):
        match = re.match(r'^(\d+)_', category)
        if match:
            idx = int(match.group(1))  # 提取第一个捕获组的数字并转为整数
            if idx>7:#暂时只提取前8类数据来分类
                continue
        datanum=0
        for file_name in os.listdir(os.path.join(dataset_path, traget_dataset,category)):
            file = os.path.join(dataset_path,traget_dataset,category, file_name)
            df = pd.read_csv(file)  # 读取信号
            x = df.iloc[:, 0:n_channels].values.astype(np.float32)  # (,)
            x_repeat = np.tile(x[:, np.newaxis], (1, 4))  # 将第一列复制四份变成 (86329, 4)这里重要

            for l_use in range(len(x)):
                if (l_use % 1 == 0) & (l_use >= 128):  
                    data_win = x_repeat[l_use - 128:l_use]  
                    data_all[idx].append(data_win)
                    label_all[idx].append(idx)

            data_split_point.append(len(data_all[idx]))#记录每个手势有多少个时间窗
        datanum_all.append(data_split_point[len(data_split_point)-1])

            # 得到data_all(分类数12，窗口数，窗口长64，通道数1)，label_all（分类数12，窗口数）

    for c in range(classes):
        data_all[c] = np.asarray(data_all[c])  # 转为numpy
        data_all[c] = data_all[c].astype('float16')  # 转为float16
        data_all[c] = np.array(data_all[c]).reshape((-1, 1, 128, 4))  # 调整形状(分类数12,窗口数，1，窗口长128，通道数1)这里重要
        label_all[c] = np.asarray(label_all[c])
        label_all[c] = label_all[c].astype('int8')
    return data_all,label_all,data_split_point,datanum_all

subject_num=1
def result_intra(x, y,datanum_all):
    x1_a_subject=[]
    x1_b_subject = []
    y1_a_subject = []
    y1_b_subject = []
    for subject in range(subject_num):
        #print('s'+str(subject))
        X1 = x[subject]  # Train 某一用户的所有数据(分类数，窗口数，1，窗口长128，通道数4)

        X1_a, X1_b, Y1_a, Y1_b = [], [], [], []
        #data_split_point_for_train = []
        for c in range(classes):
            ges_num=datanum_all[subject][c]
            X1_a.append(X1[c][:int(ges_num*0.6)])#前60%作为训练集，后40%作为测试集
            X1_b.append(X1[c][int(ges_num*0.6):])
            Y1_a.append([c]*len((X1[c][:int(ges_num*0.6)])))
            Y1_b.append([c]*len((X1[c][int(ges_num*0.6):])))

        X1_a = np.vstack(X1_a)#所有类别合并
        X1_b = np.vstack(X1_b)
        Y1_a = np.hstack(Y1_a)
        Y1_b = np.hstack(Y1_b)
        x1_a_subject.append(X1_a)
        x1_b_subject.append(X1_b)
        y1_a_subject.append(Y1_a)#
        y1_b_subject.append(Y1_b)
    return x1_a_subject,y1_a_subject,x1_b_subject,y1_b_subject#(用户数，窗口数，1，窗口长64，通道数4)
'''
输入文件，
'''
'''
subject_num = 1
x_all, y_all = [[] for _ in range(subject_num)],  [[] for _ in range(subject_num)]
data_split_point_all,datanum_all = [[] for _ in range(subject_num)],[[] for _ in range(subject_num)]
for s in range(subject_num):

    print("Loading...S" + str(s+1))
    x_all[s],y_all[s],data_split_point_all[s],datanum_all[s] = get_feature_window(s)#x_all(用户数27，分类数16，窗口数，1，窗口长32，特征数10)y_all(用户数27，分类数16，窗口数)data_split_point_all暂时用处不大，每个手势的文件数不同
    #print("d",y_all[s])
x1_a,y1_a,x1_b,y1_b=result_intra(x_all,y_all,datanum_all)
'''
