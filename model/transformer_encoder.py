import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff=2048,
        dropout=0.1
    ):

        super().__init__()

        # 创建 N 个 EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 最后的 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        # 依次通过每一层 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)

        # 最终 normalization
        x = self.norm(x)

        return x