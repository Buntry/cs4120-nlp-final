import math
import numpy as np
from keras.utils import Sequence, to_categorical
from card_dataset import CardDataset
from constants import *

# Load sequences of data from CardDataset
class CardSequence(Sequence):
    def __init__(self, card_dataset, vocab_size, seq_len=12, max_seq_len=12):
        self.card_dataset = card_dataset
        self.vocab_size = vocab_size
        self.pad_id = card_dataset.tokenizer.token_to_id(CARD_PAD)
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.card_dataset)

    def __getitem__(self, idx):
        X, y = list(), list()
        seq_ids = self.card_dataset[idx].ids
        for window_start in range(len(seq_ids) - self.seq_len):
            X.append(seq_ids[window_start:window_start+self.seq_len])
            y.append(to_categorical(seq_ids[window_start+self.seq_len], num_classes=self.vocab_size))
        return np.array(X), np.array(y)

