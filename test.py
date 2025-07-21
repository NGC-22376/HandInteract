import time
from datetime import datetime
from utils.show.blue_connection import *
from utils.show.data_process import *
from utils.show.voice_to_text import voice_to_text
from utils.show.text_to_voice import synthesize_speech
from utils.show.IMU_all2 import evl
from playsound import playsound
import torch
import threading

import sys
import os
# 添加 TTS 的“真正包目录”到 sys.path 中
sys.path.insert(0, "./TTS")
from TTS.api import TTS #虽然编辑器认为有错，但是可以成功运行，所以不要动了！


# 设备MAC地址
chip_mac = "58:56:00:01:20:44"

bt = BlueTooth(mac=chip_mac)
if not bt.connect():
    exit()

data_in_one_turn = []
flag = 1  # 是否为连接上的第一组数据
last_result=None  # 是否重复识别

def receive():
    global data_in_one_turn, bt, last_result
    while 1:
        data, is_new_data = bt.receive_data()
        # 一组数据接收完毕之后，转成csv存储
        if is_new_data:
            csv_path = '.../data/test_dir'
            to_csv(data_in_one_turn, csv_path)
            data_in_one_turn = []
            result = evl(csv_path)

            # 检查分类结果是否与上次相同，如果相同则跳过
            if result != last_result:
                print("分类结果：", result)
                last_result = result
                # 将处理后的数据通过蓝牙发送
                bt.send_msg(result)

                # 使用音色克隆模块播放分类结果
                def _play():
                    audio_file = synthesize_speech(tts, result)
                    playsound(audio_file)  # 连接蓝牙扬声器√

                threading.Thread(target=_play).start()

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
    print("初始化")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # 启动接收和发送线程
    threading.Thread(target=receive, daemon=True).start()
    threading.Thread(target=send, daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
