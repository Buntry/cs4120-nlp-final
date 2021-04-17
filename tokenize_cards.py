from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

from constants import *

def tokenize_cards(files=['./dataset/cards_train.txt', './dataset/cards_val.txt'], output_dir='./tokenizer'):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files=files, special_tokens=SPECIAL_TOKENS + OTHER_TOKENS)
    tokenizer.save_model(output_dir)

if __name__ == "__main__":
    tokenize_cards()

