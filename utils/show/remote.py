import requests

socket_result={}
CLOUD_URL='http://47.109.84.211:8080/api/data'
commands_sentence = [
    "老师们好，我是手语翻译手环团队",
    "让世界听见",
    "你好世界",
    "为了帮助聋人说话",
    "我想休息",
    "谢谢"
]
commands_word = [
    "你", "我", "大家", "好", "世界", "想", "是", "不", "说话", 
    "为了", "听见", "手语", "让", "中国", "成都", "聋人", "手语", 
    "休息", "休息", "帮助", "谢谢", "翻译", "手环"
]


'''接收socket数据并返回'''
def receive_socket_data():
    try:
        response = requests.post(CLOUD_URL,timeout=600)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[CLOUD ERROR]: {e}")
        return None
    
    try:
        decoded = result["msg"]
        print(f"[CLOUD] 收到：{decoded}")
        return decoded
    except Exception as e:
        print(f"[CLOUD ERROR] {e}")
        return None