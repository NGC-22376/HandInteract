import os
import pickle
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
from tensorflow.python.keras.utils.np_utils import *

classes = 16  # 1~8，17:静态，9~16动态，17不用
repetition = 10

def butter_lowpass(cutoff, fs, order=2):
    """
    设计巴特沃斯低通滤波器。

    参数:
    - cutoff: 截止频率 (Hz)
    - fs: 采样频率 (Hz)
    - order: 滤波器阶数

    返回:
    - b, a: 滤波器的分子和分母系数
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计巴特沃斯滤波器
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """
    应用低通滤波器到数据。

    参数:
    - data: 输入信号
    - cutoff: 截止频率 (Hz)
    - fs: 采样频率 (Hz)
    - order: 滤波器阶数

    返回:
    - y: 滤波后的信号
    """
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

# 示例使用

# 数据集Ninapro DB1(27 subjects, 17 gesturess, 10 times)
# sEMG-100Hz， 10 electrodes

# file_s:subject number(1-10);file_d:day of the acquisition(1-5);file_t:time of the acquisition(1-2)
def data_preprocessing(file_s):
    fs = 100  # 采样频率
    electrodes = 10  # 电极数
    data_all = [[] for _ in range(classes)]  # sEMG与acc合并，创建列表的列表。data_all[0] 存储手势 1 的数据，
    ges_label = [[] for _ in range(classes)] # 每个时间窗对应手势的标签

    data = scipy.io.loadmat(
        r'D:\QQdownload\NinaproDB1\S' + str(file_s) + '_A1_E2.mat')#'''不同用户的数据'''

    data_length = len(data['emg'])  # 数据长度
    sEMG = data['emg']  # sEMG信号
    stimulus = data['restimulus']  # Movement and No movement(0为NM,否则为M)'''标签'''


    fs = 100.0  # 采样频率 (Hz)
    cutoff = 1.0  # 截止频率 (Hz)
    order = 2  # 滤波器阶数
    # 滤波
    sEMG_T = [[row[i] for row in sEMG] for i in range(electrodes)]  # 转置
    sEMG_butter_T = butter_lowpass_filter(sEMG_T, cutoff, fs, order)
    sEMG_butter = [[row[i] for row in sEMG_butter_T] for i in range(data_length)]  # 转置

    data_split_point = []
    # 17分类
    ges_index = 0
    ges_index_rest = 0
    ges_index_motion = 0
    dynamic_start_point = 0
    dynamic_end_point = 0
    ges_index_save = 0
    original_len = []

    for l in range(1, data_length):
        if stimulus[l - 1][0] == 0 and stimulus[l][0] != 0:#开始动
            ges_index = ges_index_motion // repetition + 1
            ges_index_motion += 1
            if(ges_index!=stimulus[l][0]):
                print('error')
            dynamic_start_point = l
        elif stimulus[l - 1][0] != 0 and stimulus[l][0] == 0:#结束动
            dynamic_end_point = l
            ges_index_save = ges_index
            ges_index = ges_index_rest
        if (ges_index == 0)&(ges_index_save<=classes):
            if(dynamic_end_point>dynamic_start_point):
                if((ges_index_save>=1)&(ges_index_save<=8)):#静态动作
                    original_len.append(dynamic_end_point - dynamic_start_point)#记录原始动作长度
                    if dynamic_end_point - dynamic_start_point < 200:
                        data_win_use = sEMG_butter[dynamic_start_point: dynamic_end_point]#如果长度较短，直接取原始长度
                    else:
                        # 否则直接取起始到结束的所有数据
                        data_win_use = sEMG_butter[dynamic_start_point+30: dynamic_end_point-30]#如果长度较长，收缩两端点
                elif((ges_index_save>=9)&(ges_index_save<=16)):#暂时不考虑动态动作，只用前8类静态动作
                    data_win_use = sEMG_butter[dynamic_start_point+30:dynamic_end_point-30]#动态动作直接收缩两端点？为什么？
                #时间窗分割
                for l_use in range(len(data_win_use)):
                    if (l_use % 8 == 0) & (l_use >= 128):  # 时间窗5ms
                        data_win = data_win_use[l_use - 128:l_use]#对信号切片取时间窗
                        data_all[ges_index_save-1].append(data_win)#某一动作的时间窗放入该手势的列表
                        ges_label[ges_index_save-1].append(ges_index_save-1)  # 记录当前手势的标签
                data_split_point.append(len(data_all[ges_index_save-1]))#记录每个手势有多少个时间窗
                dynamic_start_point = 0
                dynamic_end_point = 0

    for c in range(classes):
        data_all[c] = np.asarray(data_all[c])#转为numpy(分类数16,窗口数，窗口长32，特征数10)
        data_all[c] = data_all[c].astype('float16')#转为float16
        data_all[c] = np.array(data_all[c]).reshape((-1, 1,128, 10))#调整形状(分类数16,窗口数，1，窗口长，特征数)
        ges_label[c] = np.asarray(ges_label[c])
        ges_label[c] = ges_label[c].astype('int8')

    return data_all, ges_label,data_split_point

x_all, y_all = [[] for _ in range(27)],  [[] for _ in range(27)]
data_split_point_all = [[] for _ in range(27)]
for s in range(27):

    print("Loading...S" + str(s+1))
    x_all[s],y_all[s],data_split_point_all[s] = data_preprocessing(s + 1)#x_all(用户数27，分类数16，窗口数，1，窗口长32，特征数10)y_all(用户数27，分类数16，窗口数)
    #print("d",y_all[s])

pickle_file = '../preprocessed_data/DB1_16c_1280ms_80ms_with_split_for_torch_new.pickle'
if not os.path.isfile(pickle_file):  # 判断是否存在此文件，若无则存储
    print('Saving data to pickle file...')
    try:
        with open('../preprocessed_data/DB1_16c_1280ms_80ms_with_split_for_torch_new.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'x': x_all,
                    'y': y_all,
                    'data_split_point_all':data_split_point_all,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')
