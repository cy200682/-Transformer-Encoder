import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):

        super().__init__()

        # 第一层线性层
        self.linear1 = nn.Linear(d_model, d_ff)

        # 激活函数
        self.relu = nn.ReLU()

        # 第二层线性层
        self.linear2 = nn.Linear(d_ff, d_model)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x shape
        # (batch , seq , d_model)

        x = self.linear1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x