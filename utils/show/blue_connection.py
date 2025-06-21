import bluetooth
from data_process import parse_raw_data


def is_sock_close(sock):
    try:
        sock.recv(1024)
        return False
    except:
        return True

def reconn_sock(sock, mac):
    sock.close()
    print("断开，尝试重连中...")
    while sock == None or is_sock_close(sock):
        sock = connect_device(mac)
    return sock

def scan_devices():
    """扫描附近的蓝牙设备"""
    print("正在扫描蓝牙设备...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    print(f"找到 {len(nearby_devices)} 个蓝牙设备")
    
    for addr, name in nearby_devices:
        print(f"\t设备名: {name}, MAC地址: {addr}")
    
    return nearby_devices

def connect_device(mac_address, port=1):
    """连接到指定MAC地址的蓝牙设备"""
    try:
        print(f"正在连接到 {mac_address}...")
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.settimeout(2)
        sock.connect((mac_address, port))
        print(f"已成功连接到 {mac_address}")
        return sock
    except ConnectionError:
        print(f"连接失败")
        return None
    
def receive_data(sock, mac, buffer_size=1024):
    """从蓝牙连接接收数据"""
    try:
        data = sock.recv(buffer_size)
        print(data)
        return data.hex(), sock

    except bluetooth.btcommon.BluetoothError as e:
        print(f"接收数据时出错: {e}")
        return None, sock
    
    except OSError:
        sock = reconn_sock(sock, mac)
        return None, sock
    
def send_msg(sock, msg, mac):
    try:
        sock.send("@" + msg + "\r\n")
        print("发送{msg}成功")
    except bluetooth.btcommon.BluetoothError as e:
        print(f"发送数据错误: {e}")
        
    except OSError:
        sock = reconn_sock(sock, mac)
        
    return sock