from tokenizers import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

from constants import *

if __name__ == "__main__":
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(
        files=['./dataset/cards_train.txt', './dataset/cards_val.txt'],
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.save_model('./tokenizer')

