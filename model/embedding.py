import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):

        # x shape
        # (batch , seq)

        out = self.embedding(x)

        # (batch , seq , d_model)

        return out