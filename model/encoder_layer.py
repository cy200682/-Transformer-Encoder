import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):

        super().__init__()

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)

        # Feed forward network
        self.ffn = FeedForward(d_model, d_ff)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # -------------------
        # 1 MultiHeadAttention
        # -------------------

        attn_out = self.mha(x, mask)

        # Residual connection
        x = x + self.dropout(attn_out)

        # LayerNorm
        x = self.norm1(x)

        # -------------------
        # 2 Feed Forward
        # -------------------

        ffn_out = self.ffn(x)

        # Residual connection
        x = x + self.dropout(ffn_out)

        # LayerNorm
        x = self.norm2(x)

        return x