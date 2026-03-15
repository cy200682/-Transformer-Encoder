import torch
from torch.utils.data import Dataset
class TextDataset(Dataset):

    def __init__(self, texts, tokenizer, max_len=10):

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode(self, text):

        # 1 tokenizer encode
        ids = self.tokenizer.encode(text)

        # 2 padding / truncation
        if len(ids) < self.max_len:
            ids += [self.tokenizer.word2idx["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids)

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        ids = self.encode(text)

        return ids

