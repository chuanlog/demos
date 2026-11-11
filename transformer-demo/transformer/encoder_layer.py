import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer.feed_forward import FeedForward
from transformer.multi_head_attention import MultiHeadAttention


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 注意力
        attn_output = self.self_attn(x, x, x, mask=mask)
        x=self.norm1(x+self.dropout1(attn_output))

        # 前馈
        ffn_output = self.ffn(x)
        x = self.norm2(x+self.dropout2(ffn_output))
        return x
