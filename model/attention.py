import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):

        # Q K V shape
        # (batch , seq , d_k)

        d_k = Q.size(-1)

        # Step1 计算 QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step2 scale
        scores = scores / math.sqrt(d_k)

        # Step3 mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step4 softmax
        attention = torch.softmax(scores, dim=-1)

        # Step5 加权 V
        output = torch.matmul(attention, V)

        return output, attention