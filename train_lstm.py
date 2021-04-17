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

model_name = "lstm-mtg"
num_epochs = 25
dropout = 0.3
verbose = True
seq_len = 10
n_hidden = 30
n_embed = 248

# load tokenizer
tokenizer = load_tokenizer()
V = tokenizer.get_vocab_size()

# load cards
cardset = CardDataset('./dataset/cards_train.txt', tokenizer, to_tensor=False)

# pad input sequences
max_seq_len = max(len(cardset[i].ids) for i in range(len(cardset)))
tokenizer.enable_padding(direction="left", pad_token=CARD_PAD, length=max_seq_len)

# data loader
card_sequence = CardSequence(cardset, V, seq_len=seq_len, max_seq_len=max_seq_len)

# build model
model = Sequential()
model.add(Embedding(V, n_embed, input_length=seq_len, mask_zero=True))
model.add(Bidirectional(LSTM(n_hidden, return_sequences=True)))
model.add(Bidirectional(LSTM(n_hidden)))
model.add(Dropout(dropout))
model.add(Dense(V, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train the model
model.fit(card_sequence, epochs=num_epochs, verbose=verbose, callbacks=[
    LogBatchLoss(f"./saved/{model_name}.log.csv", log_every_x_batches=500)
])
model.save(f"./saved/{model_name}")
