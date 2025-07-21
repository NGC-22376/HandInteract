import bluetooth


def scan_devices():
    """扫描并打印附近的蓝牙设备"""
    print("正在扫描蓝牙设备...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    print(f"找到 {len(nearby_devices)} 个蓝牙设备")

    for addr, name in nearby_devices:
        print(f"\t设备名: {name}, MAC地址: {addr}")


class BlueTooth:
    def __init__(self, mac):
        self.sock = None
        self.mac = mac

    def connect(self):
        """连接蓝牙"""
        try:
            print(f"正在连接到 {self.mac}...")
            self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.sock.settimeout(2)
            self.sock.connect((self.mac, 1))
            print(f"已成功连接到 {self.mac}")
        except ConnectionError:
            print(f"连接失败")
            return None

    def receive_data(self):
        """从蓝牙连接接收数据"""
        is_new_data = False
        try:
            header = self.sock.recv(5)  # 接收报头并打印
            print(header)
            # 检查收到的数据
            if len(header) != 5:
                raise ValueError(f"报头长度错误:\"{header.hex()}\"长度{len(header)}")
            # 检查帧头
            if header[0:2] != b'\x59\x53':
                raise ValueError(f"帧头错误:{header[0:2].hex()}")
            # 检查帧编号
            if header[3] == b'\x01':
                is_new_data = True
            # 获取数据长度，并接收
            length = int.from_bytes(header[4:], byteorder="little")
            data = self.sock.recv(length)
            # 检查校验位
            verify_bytes = self.sock.recv(2)
            checksum = (sum(header) + sum(data)) & 0xFFFF
            if checksum != int.from_bytes(verify_bytes, byteorder="little"):
                raise ValueError(f"校验和错误:收到的校验和{checksum:04X}, 计算的校验和{checksum:04X}")
            return data, is_new_data

        except ValueError as e:
            print(f"接收数据时出错: {e}")
            return None,

        except OSError as e:
            print(f"连接出错：{e}")
            return None

    def send_msg(self, msg):
        try:
            self.sock.send("@" + msg + "\r\n")
            print("发送{msg}成功")
        except bluetooth.btcommon.BluetoothError as e:
            print(f"发送数据错误: {e}")

        except OSError as e:
            print(f"连接出错：{e}")
