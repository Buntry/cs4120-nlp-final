from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from keras.callbacks import CSVLogger
from keras import Sequential
from callbacks import LogBatchLoss
from card_dataset import CardDataset
from card_sequence import CardSequence
from load_tokenizer import load_tokenizer
from constants import *
import numpy as np

# Trainer for the LSTM model
class LSTMTrainer:
    def __init__(self, model_name, train_path, n_embd=248, n_units=30, seq_len=10, dropout=0.3):
        self.model_name = model_name
        self.num_embed = n_embd
        self.num_hidden_units = n_units
        self.seq_len = seq_len
        self.dropout = dropout
        self.train_path = train_path

        # load tokenizer
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()

        # load trainset
        self.trainset = CardDataset(train_path, self.tokenizer, to_tensor=False)

        # pad input sequences
        self.max_seq_len = max(len(self.trainset[i].ids) for i in range(len(self.trainset)))
        self.tokenizer.enable_padding(direction="left", pad_token=CARD_PAD, length=self.max_seq_len)

        # data loader
        self.trainseq = CardSequence(self.trainset, self.vocab_size, seq_len=self.seq_len, max_seq_len=self.max_seq_len)

        # build model
        self.model = self.build_model()

    # Helper to build the model
    def build_model(self, summarize=True):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.num_embed, input_length=self.seq_len, mask_zero=True))
        model.add(Bidirectional(LSTM(self.num_hidden_units, return_sequences=True)))
        model.add(Bidirectional(LSTM(self.num_hidden_units)))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.vocab_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if summarize:
            model.summary()
        return model
        
    # train function
    def train(self, num_epochs=25, verbose=True, log_every_x_batches=500):
        self.model.fit(self.trainseq, epochs=num_epochs, verbose=verbose, callbacks=[
            LogBatchLoss(f"./saved/{self.model_name}.log.csv", log_every_x_batches=log_every_x_batches)
        ])
        self.model.save(f"./saved/{self.model_name}")

if __name__ == "__main__":
    trainer = LSTMTrainer("lstm-mtg", "./dataset/cards_train.txt")
    trainer.train()