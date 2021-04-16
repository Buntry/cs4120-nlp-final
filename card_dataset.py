import torch
from torch.utils.data import Dataset

class CardDataset(Dataset):
    def __init__(self, card_path, tokenizer, to_tensor=True):
        with open(card_path, "r", encoding="utf-8") as card_file:
            self.cards = list(card_file.readlines())
        self.tokenizer = tokenizer
        self.to_tensor = to_tensor
    
    def __len__(self):
        return len(self.cards)
    
    def __getitem__(self, idx):
        if self.to_tensor:
            return torch.tensor(self.tokenizer.encode(self.cards[idx]))
        return self.tokenizer.encode(self.cards[idx])