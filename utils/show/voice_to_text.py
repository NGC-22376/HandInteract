import time
from pynput import keyboard
import threading
import time
import hashlib
import base64
import json
import threading
import websocket
import pyaudio
import urllib.parse
from email.utils import formatdate
import hmac
import ssl
import keyboard  # 新增：用于监听按键
from pynput import keyboard

data_remain = []
flag = 1  # 是否为连接上的第一组数据
sock = None
mac = None
is_recording = False
recognizer = None
recording_thread = None
is_pressed = False

# ===== 讯飞API参数 =====
APPID = "f765a2b2"
APIKey = "41a1eb407970fe0af8aae2507dcf4cb3"
APISecret = "MTFkMWRmMTFmYWIwNmM2ZjMxZmU2OGNk"
HOST = "ws-api.xfyun.cn"
ENDPOINT = "/v2/iat"
# 音频参数
RATE = 16000
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1


# ===== 鉴权URL生成 =====
def create_url():
    date = formatdate(usegmt=True)
    signature_origin = f"host: {HOST}\ndate: {date}\nGET {ENDPOINT} HTTP/1.1"
    signature_sha = hmac.new(APISecret.encode("utf-8"), signature_origin.encode("utf-8"),
                             digestmod=hashlib.sha256).digest()
    signature = base64.b64encode(signature_sha).decode("utf-8")

    authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    params = {"authorization": authorization, "date": date, "host": HOST}
    return f"wss://{HOST}{ENDPOINT}?" + urllib.parse.urlencode(params)


class XFRecognizer:
    def __init__(self):
        self.ws = None
        self.status = 0
        self.stop_signal = threading.Event()
        self.ws_ready = threading.Event()

    def on_message(self, ws, message, send_queue):
        data = json.loads(message)
        if data.get("code") != 0:
            print("识别错误:", data.get("message"))
        else:
            result = data['data']['result']
            if 'ws' in result:
                text = ''.join([w['cw'][0]['w'] for w in result['ws']])
                print("[识别结果]", text)
                # 将识别结果放入发送队列
                send_queue.put(text)

    def on_error(self, ws, error):
        print("WebSocket错误:", error)

    def on_close(self, ws, code, msg):
        print("WebSocket关闭")

    def on_open(self, ws):
        def run():
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("开始讲话，松开音量键结束：")

            self.ws_ready.set()  # ✅ 告诉外部“WebSocket已准备好”

            try:
                while not self.stop_signal.is_set():
                    buf = stream.read(CHUNK)
                    if not buf:
                        continue
                    d = {
                        "common": {"app_id": APPID},
                        "business": {
                            "language": "zh_cn",
                            "domain": "iat",
                            "accent": "mandarin",
                            "vad_eos": 300
                        },
                        "data": {
                            "status": self.status,
                            "format": "audio/L16;rate=16000",
                            "audio": base64.b64encode(buf).decode('utf-8'),
                            "encoding": "raw"
                        }
                    }
                    ws.send(json.dumps(d))
                    self.status = 1
                    time.sleep(0.05)
            except Exception as e:
                print("音频采集异常：", e)
            finally:
                try:
                    ws.send(json.dumps({
                        "data": {
                            "status": 2,
                            "format": "audio/L16;rate=16000",
                            "audio": "",
                            "encoding": "raw"
                        }
                    }))
                except Exception as e:
                    print("发送结束帧失败：", e)
                time.sleep(1)
                stream.stop_stream()
                stream.close()
                p.terminate()
                ws.close()

        threading.Thread(target=run).start()

    def start(self):
        self.status = 0
        self.stop_signal.clear()
        self.ws_ready.clear()
        ws_url = create_url()
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        threading.Thread(target=lambda: self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})).start()

        if self.ws_ready.wait(timeout=5):
            return True  # ✅ 表示准备完成
        else:
            print("WebSocket连接超时！")
            return False


def toggle_recording(send_queue):
    global is_recording, recording_thread, recognizer

    if not is_recording:
        recognizer = XFRecognizer(send_queue)
        recording_thread = threading.Thread(target=recognizer.start)
        recording_thread.start()
        time.sleep(0.1)  # 给线程一点时间启动

        # 等待 WebSocket 准备好再提示讲话
        if recognizer.ws_ready.wait(timeout=5):
            print("【开始录音】请讲话...")
            is_recording = True
        else:
            print("连接失败，无法开始录音。")
    else:
        print("【录音结束】正在识别...")
        is_recording = False
        if recognizer:
            recognizer.stop_signal.set()


def on_press(key, send_queue):
    if key == keyboard.Key.media_volume_up:  # 这里有点类似于直接连电脑的音量键了，所以可能不用再显式连接按钮的蓝牙
        print("🎯 蓝牙按钮按下（音量加）")
        toggle_recording(send_queue)


def voice_to_text(microphone_mac, send_queue):
    print("线程启动")
    print("按 Space 开始/停止录音，按 ESC 退出")
    with keyboard.Listener(send_queue, on_press=on_press) as listener:
        listener.join()
    print("程序已退出")
