import os

import numpy as np
import pandas as pd


def parse_raw_data(raw_data, datas):
    """处理从IMU传来的数据"""
    results = []
    error_message = []

    # 处理当前波次的数据
    # 截断
    A = raw_data[0:14]
    W = raw_data[12:24]
    B = raw_data[24:36]
    O = raw_data[36:48]
    q = raw_data[48:64]

    # 处理为int,并入结果列表
    if A[0] != b'\x10':
        error_message.append("加速度值错误")
        results += [None] * 3
    else:
        Ax = int.from_bytes(A[2:6], byteorder='little', signed=True)
        Ay = int.from_bytes(A[6:10], byteorder='little', signed=True)
        Az = int.from_bytes(A[10:14], byteorder='little', signed=True)
        results += np.array([Ax, Ay, Az]) * 1e-6

    if W[0] != b'\x20':
        error_message.append("角速度值错误")
        results += [None] * 3
    else:
        Wx = int.from_bytes(W[2:6], byteorder='little', signed=True)
        Wy = int.from_bytes(W[6:10], byteorder='little', signed=True)
        Wz = int.from_bytes(W[10:14], byteorder='little', signed=True)
        results += np.array([Wx, Wy, Wz]) * 1e-6

    if B[0] != b'\x30':
        error_message.append("磁场归一化值错误")
        results += [None] * 3
    else:
        Bx = int.from_bytes(B[2:6], byteorder='little', signed=True)
        By = int.from_bytes(B[6:10], byteorder='little', signed=True)
        Bz = int.from_bytes(B[10:14], byteorder='little', signed=True)
        results += np.array([Bx, By, Bz]) * 1e-6

    if O[0] != b'\x40':
        error_message.append("欧拉角值错误")
        results += [None] * 3
    else:
        Ox = int.from_bytes(O[2:6], byteorder='little', signed=True)
        Oy = int.from_bytes(O[6:10], byteorder='little', signed=True)
        Oz = int.from_bytes(O[10:14], byteorder='little', signed=True)
        results += np.array([Ox, Oy, Oz]) * 1e-6

    if q[0] == b'\x41':
        error_message.append("四元数值错误")
        results += [None] * 4
    else:
        q1 = int.from_bytes(q[2:6], byteorder='little', signed=True)
        q2 = int.from_bytes(q[6:10], byteorder='little', signed=True)
        q3 = int.from_bytes(q[10:14], byteorder='little', signed=True)
        q4 = int.from_bytes(q[14:18], byteorder='little', signed=True)
        results += np.array([q1, q2, q3, q4]) * 1e-6

    datas.append(results)

    return datas, error_message


def to_csv(array, file_dir):
    """将一组IMU数据保存为csv文件"""
    # 验证数组形状
    if len(array.shape) != 2 or array.shape[1] != 16:
        raise ValueError("数组必须是二维的，并且第二维大小为16")

    # 创建行索引和列索引
    T = array.shape[0]
    row_indices = [f'T{i}' for i in range(T)]
    col_indices = ['Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Bx', 'By', 'Bz', 'O1', 'O2', 'O3', 'q1', 'q2', 'q3',
                   'q4']  # 加速度,角速度,欧拉角,四元数

    # 创建DataFrame
    df = pd.DataFrame(array, index=row_indices, columns=col_indices)

    # 保存为CSV文件
    file_path = os.path.join(file_dir, f'{len(os.listdir(file_dir))}.csv')
    df.to_csv(file_path)
    print(f"CSV文件已保存至: {file_path}")

    return df
