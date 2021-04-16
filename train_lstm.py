from tokenizers import ByteLevelBPETokenizer
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from keras import Sequential
from card_dataset import CardDataset
from card_sequence import CardSequence
from constants import *

model_name = "lstm0"
num_epochs = 5
dropout = 0.3
verbose = True

# load tokenizer
tokenizer = ByteLevelBPETokenizer('./tokenizer/vocab.json', './tokenizer/merges.txt')
V = tokenizer.get_vocab_size()

# load cards
cardset = CardDataset('./dataset/cards_train.txt', tokenizer, to_tensor=False)
max_seq_len = max(len(cardset[idx]) for idx in range(len(cardset)))

# enable padding
tokenizer.enable_padding(direction='left', pad_token=CARD_PAD, pad_to_multiple_of=max_seq_len)

# get training data
card_sequence = CardSequence(cardset, V, batch_size=12)

# build model
model = Sequential()
model.add(Embedding(tokenizer.get_vocab_size(), 128, input_length=max_seq_len))
model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Bidirectional(LSTM(10, return_sequences=True)))
model.add(Dropout(dropout))
model.add(Dense(tokenizer.get_vocab_size(), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# train the model
model.fit(card_sequence, epochs=num_epochs, verbose=verbose)
model.save(f"./saved/{model_name}")