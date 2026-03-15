import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        # 创建位置编码矩阵
        # shape (max_len , d_model)

        pe = torch.zeros(max_len, d_model)

        # 位置索引
        position = torch.arange(0, max_len).unsqueeze(1)

        # 计算分母部分
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        # 偶数维
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不是参数）
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x shape
        # (batch_size , seq_len , d_model)

        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return x