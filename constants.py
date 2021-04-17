import torch

# Constants
CARD_BEGIN = "<s>"
CARD_END = "</s>"
UNK = "<unk>"
CARD_MASK = "<mask>"
CARD_PAD = "<pad>"
LF = "<lf>"
EM = "<em>"
ACT = ":"

CARDNAME = "CARDNAME"

SPECIAL_TOKENS = [CARD_PAD, CARD_BEGIN, CARD_END, UNK, EM, ACT, LF, CARD_MASK, CARDNAME]
OTHER_TOKENS = [
    "{w}", "{u}", "{b}", "{r}", "{g}", "{t}", "{s}", "{c}",
    "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}", "{9}", "{10}"
]

VAL_SPLIT = 0.2
MASK_PROB = 0.2

VAL_SPLIT = 0.05
MASK_PROB = 0.2

TRAIN_PATH = "./dataset/cards_train.txt"
VAL_PATH = "./dataset/cards_path.txt"

SEED = 123