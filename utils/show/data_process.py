def to_decimal(hex_num):
    return int.from_bytes(hex_num, byteorder='big')


def parse_raw_data(raw_data, data_remain):
    results = [] # 存放结果的列表
    error_message = []

    # 提取十六进制数据
    # 查看上一组的剩余数据
    if data_remain:# 有剩余数据 
        # 检查数据长度是否正确，每组数据应为两位的16进制数
        data_in_part = data_remain.copy()
        data_remain=[]
    else:
        data_in_part = []
    # 添加目前的数据
    data_in_part = [raw_data[i:i+2] for i in range(0, len(raw_data), 2)]
    # 将不成组的数据放进data_remain里面
    if len(data_in_part) % 36 != 0:
        data_remain = data_in_part[(36 * (len(data_in_part) // 36)):]

    # 处理当前波次的数据
    # 按每36组划分数据
    for i in range(0, len(data_in_part), 36):
        chunk = data_in_part[i:i + 36]

        # 校验位检查
        checksum = chunk[0:4]
        expected_checksum = ['01', '00', '00', '01']
        if checksum != expected_checksum:
            error_message.append("校验位不匹配")
            continue

        # 提取四通道数据（8位）
        channel_data = chunk[4:12]

        # 检查间隔位1
        separator1 = chunk[12:16]
        expected_separator1 = ["ab", "ba", "fc", "cf"]
        if separator1 != expected_separator1:
            error_message.append("间隔位1不匹配")
            continue

        # 提取加速度传感器数据（6位）
        accel_data = chunk[16:22]

        # 检查间隔位2
        separator2 = chunk[22:26]
        expected_separator2 = ["ba", "ab", "cf", "fc"]
        if separator2 != expected_separator2:
            error_message.append("间隔位2不匹配")
            continue

        # 提取陀螺仪数据（6位）
        gyro_data = chunk[26:32]

        # 检查终止位
        terminator = chunk[32:36]
        expected_terminator = ["00", "01", "01", "00"]
        if terminator != expected_terminator:
            error_message.append("终止位不匹配")
            continue

        # 处理四通道数据
        u_values = []
        for i in range(0, 8, 2):
            high = channel_data[i]
            low = channel_data[i + 1]
            combined = high + low
            decimal_value = int(combined, 16)
            # 电压计算：0-4095 对应 0-3.3V
            voltage = (decimal_value / 4095) * 3.3
            u_values.append(voltage)

        # 处理加速度传感器数据
        a_values = []
        for i in range(0, 6, 2):
            high = accel_data[i]
            low = accel_data[i + 1]
            combined = high + low
            # 补码转换
            decimal_value = int(combined, 16)
            if decimal_value > 32767:
                decimal_value -= 65536
            # 加速度计：-32768 to 32767 对应 ±16g
            g_force = (decimal_value / 32767) * 16
            a_values.append(g_force)

        # 处理陀螺仪数据
        g_values = []
        for i in range(0, 6, 2):
            high = gyro_data[i]
            low = gyro_data[i + 1]
            combined = high + low
            # 补码转换
            decimal_value = int(combined, 16)
            if decimal_value > 32767:
                decimal_value -= 65536
            # 陀螺仪：-32768 to 32767 对应 ±2000°/s
            degree_per_sec = (decimal_value / 32767) * 2000
            g_values.append(degree_per_sec)

        # 添加结果
        results.append({
            "U1": u_values[0],
            "U2": u_values[1],
            "U3": u_values[2],
            "U4": u_values[3],
            "Ax": a_values[0],
            "Ay": a_values[1],
            "Az": a_values[2],
            "Gx": g_values[0],
            "Gy": g_values[1],
            "Gz": g_values[2]
        })

    if not results:
        error_message.append("未找到有效数据")

    return data_remain, results, error_message