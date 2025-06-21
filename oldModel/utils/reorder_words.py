import os

from qianfan import Qianfan


def reorder(words):
    """
    还原手语语序为口语语序
    :param words: 手语语序序列
    :return: 口语语序句子
    """
    client = Qianfan(
        access_key=os.environ["QIANFAN_ACCESS_KEY"],
        secret_key=os.environ["QIANFAN_SECRET_KEY"]
    )

    completion = client.chat.completions.create(
        model="ernie-3.5-8k",  # 指定特定模型
        messages=[
            {'role': 'system', 'content': '你是一个手语专家'},
            {'role': 'user', 'content': f'还原手语语序{words}，仅输出口语语序下句子'}
        ]
    )

    return completion.choices[0].message.content
