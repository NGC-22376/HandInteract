from utils.filter import signal_filter
from utils.reorder_words import reorder
from utils.text_to_voice import text_to_voice, to_wav
# 获取输入数据并滤波
data_path = ""
filtered_signal = signal_filter(data_path)

# 通过网络得到信号中包含的所有手势名
words = model(filtered_signal)

# 还原为正常语序
sequence = reorder(words)

# 语音化正常语序，参考音色放入voices文件夹命名为input.xxx即可，按照格式修改下面的路径
to_wav("./voices/input.m4a")
text_to_voice(sequence, "./voices", "./voices/input.wav")