import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 前馈层
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """

        :param x: batch_size,seq_len,d_model
        :return: x 映射到高维度，激活，映射回来,处理单个词内部不同维度的关系
        """
        x=self.linear1(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x
        