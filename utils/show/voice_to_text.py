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
    signature_sha = hmac.new(APISecret.encode("utf-8"), signature_origin.encode("utf-8"), digestmod=hashlib.sha256).digest()
    signature = base64.b64encode(signature_sha).decode("utf-8")

    authorization_origin = f'api_key="{APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    params = {"authorization": authorization, "date": date, "host": HOST}
    return f"wss://{HOST}{ENDPOINT}?" + urllib.parse.urlencode(params)

# ===== 语音识别类 =====
class XFRecognizer:
    def __init__(self):
        self.ws = None
        self.status = 0
        self.stop_signal = threading.Event()
        self.ws_ready = threading.Event()
        self.result = None
        

    def on_message(self, ws, message):
        data = json.loads(message)
        if data.get("code") != 0:
            print("识别错误:", data.get("message"))
        else:
            result = data['data']['result']
            if 'ws' in result:
                text = ''.join([w['cw'][0]['w'] for w in result['ws']])
                self.result = text
                print("[识别结果]", text)

    def on_error(self, ws, error):
        print("WebSocket错误:", error)

    def on_close(self, ws, code, msg):
        print("WebSocket关闭")

    def on_open(self, ws):
        def run():
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("开始讲话，松开 space 结束：")

            self.ws_ready.set()  # 告诉外部“WebSocket已准备好”

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
            return True  #  表示准备完成
        else:
            print("WebSocket连接超时！")
            return False
