import torch
import torch.nn as nn

from .attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q K V projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # attention
        self.attention = ScaledDotProductAttention()

        # output linear
        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x):

        batch_size, seq_len, d_model = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        x = x.transpose(1, 2)

        return x

    def concat_heads(self, x):

        batch_size, num_heads, seq_len, d_k = x.size()

        x = x.transpose(1, 2)

        x = x.contiguous().view(batch_size, seq_len, self.d_model)

        return x

    def forward(self, x, mask=None):

        batch_size = x.size(0)

        # Linear projection
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # attention
        out, attn = self.attention(Q, K, V, mask)

        # concat heads
        out = self.concat_heads(out)

        # final linear
        out = self.fc(out)

        return out