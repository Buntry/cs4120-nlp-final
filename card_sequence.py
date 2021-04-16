import math
import numpy as np
from keras.utils import Sequence, to_categorical
from card_dataset import CardDataset

class CardSequence(Sequence):
    def __init__(self, card_dataset, vocab_size, batch_size=64):
        self.card_dataset = card_dataset
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def __len__(self):
        return math.ceil(len(self.card_dataset) / self.batch_size)

    def __getitem__(self, idx):
        X, y = list(), list()
        for i in range(idx * self.batch_size, min((idx+1) * self.batch_size, len(self.card_dataset))):
            seq_ids = self.card_dataset[i].ids
            X.append(seq_ids)

            target_ids = seq_ids[1:] + [0]
            y.append(to_categorical(target_ids, num_classes=self.vocab_size))

        return np.array(X), np.array(y)

