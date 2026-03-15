import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_encoder import TransformerEncoder


class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        max_len=100,
        dropout=0.1
    ):

        super().__init__()

        # 词向量
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # x shape
        # (batch , seq)

        # embedding
        x = self.embedding(x)

        # (batch , seq , d_model)

        # positional encoding
        x = self.positional_encoding(x)

        x = self.dropout(x)

        # transformer encoder
        x = self.encoder(x, mask)

        # output layer
        logits = self.fc_out(x)

        return logits