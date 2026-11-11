## 多头注意力层
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力（Multi-Head Attention）。

    把 d_model 维的输入拆成 n_head 个"头"并行做缩放点积注意力，
    每个头只在 d_k = d_model / n_head 维的子空间里看输入，
    最后把所有头的结果拼回 d_model 维，再经过一次线性变换 W_o。

    参数:
        d_model (int): 模型/隐藏层维度，也是输入、输出的最后一维大小
        n_head  (int): 注意力头的数量，必须能整除 d_model
        dropout (float): 作用在注意力权重上的 dropout 概率，默认 0.1
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.n_head = n_head  # 头的数量
        self.d_k = d_model // n_head  # k矩阵的维度（每个头分到的维度）

        # 校验：d_model 必须能被 n_head 整除，否则拆分后维度对不上
        assert (self.d_k * n_head == d_model), f"d_model {d_model} not divisible by n_heads {n_head}"

        # 定义QKVO
        # 4 个不带偏置的全连接层，相当于 4 个可学习的权重矩阵：
        # W_q / W_k / W_v 把输入映射成 Query / Key / Value
        # W_o 把拼接后的多头输出再融合一次
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        """单次缩放点积注意力：softmax(QK^T / sqrt(d_k)) V。

        softmax 本质上是把每一行归一化成概率分布，
        意思是"当前位置要把多少注意力分给其他每个位置"。

        参数:
            Q: (batch_size, n_heads, seq_len_q, d_k) 查询矩阵
            K: (batch_size, n_heads, seq_len_k, d_k) 键矩阵
            V: (batch_size, n_heads, seq_len_k, d_k) 值矩阵，长度与 K 一致
            mask: 形状需能广播到 (batch_size, n_heads, seq_len_q, seq_len_k)，
                  1 表示可见、0 表示屏蔽（填充位或未来位置）
        返回:
            (batch_size, n_heads, seq_len_q, d_k) 加权求和后的值
        """
        # Q K V: (batch_size, n_heads, seq_len, d_k)

        # 计算注意力得分 Q @ K^T：每个 query 和所有 key 做点积，得到"相似度"
        # K.transpose(-2, -1) 交换最后两维 -> (batch_size, n_heads, d_k, seq_len_k)
        # 除以 sqrt(d_k) 是缩放，防止 d_k 变大时点积过大、softmax 梯度消失
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)

        # 处理mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将mask为0的地方填充极小值 -1e9

        # 在最后一维（key 方向）做 softmax，使每个 query 的注意力权重和为 1
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)  # 对注意力权重进行dropout
        # 用权重对 V 做加权求和：权重 (…, seq_len_q, seq_len_k) @ V (…, seq_len_k, d_k)
        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, d_k)
        return output

    def forward(self, Q, K, V, mask=None):
        """前向传播。

        参数:
            Q: (batch_size, seq_len_q, d_model) 查询序列（自注意力时就是输入 x）
            K: (batch_size, seq_len_k, d_model) 键序列
            V: (batch_size, seq_len_k, d_model) 值序列
            mask: 注意力掩码，一般形状为 (batch_size, 1, 1, seq_len_k)
                  或 (batch_size, 1, seq_len_q, seq_len_k)，1 可见、0 屏蔽
        返回:
            (batch_size, seq_len_q, d_model) 注意力输出
        """
        batch_size = Q.size(0)
        # 将d_model维拆成n_head头，每一个头d_k
        # 三步：
        #   1) 线性层把最后一维映射成 d_model（可学习投影）
        #   2) view 把 d_model 拆成 (n_head, d_k)
        #   3) transpose 把"头"这一维换到 seq_len 前面，变成 (batch, n_head, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        attn_output = self.scale_dot_product_attention(Q, K, V, mask)# 输出 (batch_size,n_head,seq_len,d_k)

        # 多头输出拼接回d_model维度
        # transpose(1, 2) 把 (batch, n_head, seq_len, d_k) 变回 (batch, seq_len, n_head, d_k)
        # contiguous() 让内存连续，view 才能安全地合并最后两维 -> (batch, seq_len, d_model)
        attn_output=attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        output = self.W_o(attn_output)  # 融合各头信息，输出维度仍为 d_model
        return output # 输出 batch_size , seq_len , d_model

