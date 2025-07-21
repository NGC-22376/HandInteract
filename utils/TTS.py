import os
from TTS.api import TTS

# 初始化模型（可放在模块加载时，只需加载一次）
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

def synthesize_voice(text: str, speaker_wav_path: str, output_path: str, language: str = "zh"):
    """
    使用 XTTS v2 合成语音（支持音色克隆）

    参数:
        text (str): 要合成的文本
        speaker_wav_path (str): 参考音频路径（用于音色克隆）
        output_path (str): 输出 WAV 文件路径
        language (str): 语言代码，默认 "zh"（中文）
    """
    if not os.path.isfile(speaker_wav_path):
        raise FileNotFoundError(f"参考音频文件不存在: {speaker_wav_path}")

    print(f"[INFO] 正在合成文本: {text}")
    print(f"[INFO] 使用参考音频: {speaker_wav_path}")
    print(f"[INFO] 输出路径: {output_path}")

    tts_model.tts_to_file(
        text=text,
        speaker_wav=speaker_wav_path,
        language=language,
        file_path=output_path
    )

synthesize_voice(
    text="你好，欢迎来到语音克隆测试。",
    speaker_wav_path="你好.wav",
    output_path="output5.wav"
)