from utils.filter import signal_filter
from utils.reorder_words import reorder

# 获取输入数据并滤波
data_path = ""
signal_filter(data_path)

# 通过网络得到信号中包含的所有手势名
words = model()

# 还原为正常语序
sequence = reorder(words)

# 语音化正常语序
broadcast(sequence)