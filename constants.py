# Constants
CARD_BEGIN = "<s>"
CARD_END = "</s>"
UNK = "<unk>"
CARD_MASK = "<mask>"
CARD_PAD = "<pad>"
LF = "<lf>"

CARDNAME = "CARDNAME"

SPECIAL_TOKENS = [CARD_BEGIN, CARD_END, UNK, LF, CARD_MASK, CARDNAME]

VAL_SPLIT = 0.2
MASK_PROB = 0.2

CARDS_PATH = "./dataset/cards.txt"

SEED = 123