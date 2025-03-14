import os

import pandas as pd
from nptdms import TdmsFile


def into_csv(ori_dir_path, csv_dir_path, channel_num=4, dt=4e-5, delete_ori=False):
    """
    从tdms文件里批量获取数据，并存成csv格式
    :param ori_dir_path: tdms文件的上一级目录
    :param csv_dir_path: csv文件保存位置的上一级目录
    :param channel_num: 采集通道数，默认为4通道
    :param dt: 采样间隔，默认为4e-5s
    :param delete_ori: 是否删除原始tdms文件，默认为否
    :return: null
    """
    for file_name in os.listdir(ori_dir_path):
        file = os.path.join(ori_dir_path, file_name)
        if os.path.splitext(file)[1][1:] == "tdms":
            tdms_file = TdmsFile.read(file)  # 读取TDMS文件
            all_groups = tdms_file.groups()  # 获取所有组名
            group_name = all_groups[1].name  # 解析发现目标数据在第二组，且组名与DAQExpress中的记录名保持一致
            group = tdms_file[group_name]  # 获取该组
            channel = group.channels()  # 获取所有通道
            datas = {}
            # 解析每个通道数据
            for i in range(0, channel_num):
                current_channel = group[channel[i].name]
                voltage_data = current_channel.data
                if i == 0:
                    datas["time"] = [i * dt for i in range(0, len(voltage_data))]
                datas[channel[i].name] = voltage_data

            # 保存为csv文件
            df = pd.DataFrame(datas)
            df.to_csv(os.path.join(csv_dir_path, group_name + ".csv"), index=False, encoding='utf-8-sig')

            # 删除原始tdms文件及对应的index文件
            if delete_ori:
                os.remove(file)
                os.remove(os.path.splitext(file)[0] + '.tdms_index')
