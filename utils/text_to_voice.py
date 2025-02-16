import os

import torch
from melo.api import TTS
from pydub import AudioSegment

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


def to_wav(input_path):
    """
    转换音频格式为.wav
    :param input_path: 要转换的音频文件
    :return:
    """
    if not os.path.exists(input_path):
        print(f"音频文件{input_path}不存在")
        exit(-1)
    dir_path = os.path.dirname(input_path)

    # 加载源文件
    audio = AudioSegment.from_file(input_path)

    # 转换并保存为 .wav 文件
    base_name = os.path.basename(input_path)
    audio.export(os.path.join(dir_path, os.path.splitext(base_name)[0] + '.wav'), format="wav")


def text_to_voice(text, output_dir, refer_voice):
    """
    根据模仿音色，将文字输出为音频
    :param text: 要输出的文字
    :param output_dir: 输出的音频所在文件夹
    :param refer_voice: 要模仿的音色
    :return:
    """
    openvoice_path = "../openvoice"
    ckpt_converter = f'{openvoice_path}/checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)
    reference_speaker = refer_voice  # This is the voice you want to clone
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

    texts = {
        'ZH': text
    }

    src_path = f'{output_dir}/tmp.wav'

    speed = 1.0

    for language, text in texts.items():
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

            source_se = torch.load(f'{openvoice_path}/checkpoints_v2/base_speakers/ses/{speaker_key}.pth',
                                   map_location=device)
            model.tts_to_file(text, speaker_id, src_path, speed=speed)
            save_path = f'{output_dir}/result_{speaker_key}.wav'

            # Run the tone color converter
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message=encode_message)
