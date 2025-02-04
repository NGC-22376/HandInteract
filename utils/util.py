from datetime import datetime

def print_msg(msg):
    t = datetime.now()
    print(f"{t.strftime('%Y-%m-%d %H:%M:%S')}-{msg}")