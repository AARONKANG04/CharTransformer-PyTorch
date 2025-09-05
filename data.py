import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = { ch : i for i, ch in enumerate(self.vocab)}
        self.itos = { i : ch for i, ch in enumerate(self.vocab)}
        self.tokens = [self.stoi[ch] for ch in text]

    def __len__(self):
        return len(self.tokens) - self.max_seq_len

    def __getitem__(self, idx):
        assert idx < len(self), "index must be within the length of the dataset."
        data = torch.tensor(self.tokens[idx:idx+self.max_seq_len], dtype=torch.long)
        target = torch.tensor(self.tokens[idx+1:idx+self.max_seq_len+1], dtype=torch.long)
        return data, target


