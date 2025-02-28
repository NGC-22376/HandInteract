import time

import winsound


def beep_every_4_seconds():
    for i in range(0, 20):
        time.sleep(1.5)
        winsound.Beep(1000, 500)  # 1000 Hz频率，500毫秒持续时间
        time.sleep(1.5)


# 每隔4秒重复一次

if __name__ == "__main__":
    beep_every_4_seconds()
    winsound.Beep(2000, 500)
