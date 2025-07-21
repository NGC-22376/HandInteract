import soundfile as sf
import numpy as np
import torchaudio


def synthesize_speech(tts,text,filename="output.wav"):
  def preprocess_wav(path, out_path="clean.wav"):
    #若非单声道或采样率不匹配，调整为适合模型的输入
    signal, sr = torchaudio.load(path)
    if signal.shape[0] > 1:
      signal = signal.mean(dim=0, keepdim=True)
    if sr != 16000:
      resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
      signal = resample(signal)
    torchaudio.save(out_path, signal, 16000)
    return out_path

  speaker_wav = preprocess_wav("speaker_voice.wav")

# TTS with list of amplitude values as output, clone the voice from `speaker_wav`
  wav = tts.tts(
    text=text,
    speaker_wav=speaker_wav,
    language="zh"
  )

  # 保存为 WAV 文件
  # 先转成 numpy array
  wav_np = np.array(wav)
  # 保存为 16-bit PCM WAV 文件
  sf.write(filename, wav_np, samplerate=24000)
  return filename
