import torch
import torch.nn as nn


class TransformerLoss:

    def __init__(self, pad_idx=0):

        # 忽略 padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def __call__(self, logits, targets):

        # logits
        # (batch , seq , vocab)

        # targets
        # (batch , seq)

        vocab_size = logits.size(-1)

        # reshape
        logits = logits.view(-1, vocab_size)

        targets = targets.view(-1)

        loss = self.criterion(logits, targets)

        return loss