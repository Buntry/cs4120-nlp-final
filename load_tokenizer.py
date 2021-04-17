from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from constants import *

# Loads a tokenizer from a vocab/merges file
def load_tokenizer(vocab='./tokenizer/vocab.json', merges='./tokenizer/merges.txt', gpt=False, load_from=None):
    if gpt:
        if load_from:
            tokenizer = GPT2Tokenizer.from_pretrained(load_from)
        else:
            tokenizer = GPT2Tokenizer(
                vocab, merges, 
                bos_token=CARD_BEGIN, eos_token=CARD_END, sep_token=CARD_END,
                unk_token=UNK, pad_token=CARD_PAD, mask_token=CARD_MASK, padding_side="left"
            )
    else:
        tokenizer = ByteLevelBPETokenizer(vocab, merges)
        tokenizer.add_special_tokens(SPECIAL_TOKENS + OTHER_TOKENS)
        tokenizer.mask_token = CARD_MASK
    
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer
