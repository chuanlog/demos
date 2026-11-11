import torch.nn as nn
from transformer.encoder_layer import EncoderLayer

# 编码器，多个编码器层堆叠
class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_layers, drop_out=0.1):
        super().__init__()
        # n个编码器层堆叠
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, drop_out) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: batch_size seq_len d_model
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x) # 最后做一次归一化
        return x # (batch_size, seq_len, d_model)



