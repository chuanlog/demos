# 正弦位置编码
import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    """把位置信息加到词向量上：偶数维用 sin，奇数维用 cos。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000):
        # d_model: 词向量维度；max_len: 最大序列长度
        super().__init__()

        # 位置下标 0..max_len-1，形状 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 每个维度对应的频率，形状 (d_model/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # pe 形状 (1, max_len, d_model)，第 0 维留给 batch 做广播
        pe = torch.zeros(1, max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        self.pe = pe

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # self.pe[:, :seq_len]: (1, seq_len, d_model)，广播相加，输出形状不变
        x = x + self.pe[:, :x.size(1)]
        return x