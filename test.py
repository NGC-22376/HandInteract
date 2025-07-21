import time
from datetime import datetime
from utils.show.blue_connection import *
from utils.show.data_process import *
from utils.show.voice_to_text import voice_to_text
import threading

# 设备MAC地址
chip_mac = "58:56:00:01:20:44"

bt = BlueTooth(mac=chip_mac)
if not bt.connect():
    exit()

data_in_one_turn = []
flag = 1  # 是否为连接上的第一组数据

def receive():
    global data_in_one_turn, bt
    while 1:
        data, is_new_data = bt.receive_data()
        # 一组数据接收完毕之后，转成csv存储
        if is_new_data:
            to_csv(data_in_one_turn, '.../data/test_dir')
            data_in_one_turn = []
        # 解析数据
        if data:
            data_in_one_turn, errors = parse_raw_data(data, data_in_one_turn)
            print(f"[{datetime.now().strftime(r'%m.%d %H:%M:%S')}]接收数据时出错：{errors}\n")

        time.sleep(0.2)


def send():
    global bt
    while 1:
        msg = voice_to_text()
        bt.send_msg(msg)


def main():
    # 启动接收和发送线程
    threading.Thread(target=receive, daemon=True).start()
    threading.Thread(target=send, daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
