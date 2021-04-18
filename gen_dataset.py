from constants import *
import re, json, argparse
from sklearn.model_selection import train_test_split

# load data entry from card json
def load_card_data(card_json_path):
    with open(card_json_path, "r", encoding="utf-8") as cards_file:
        cards_db = json.load(cards_file)
        return cards_db['data']

# restrictions for cards to look at
def passes_restrictions(card_info):
    # only consider cards that are vintage legal
    if ("vintage" not in card_info['legalities'] or card_info['legalities']['vintage'] != 'Legal'):
        return False
    # remove empty-texted cards ("vanilla creatures")
    elif 'text' not in card_info or not card_info['text'].strip():
        return False
    return True

# card information to card document
def card_info_to_card_doc(card_name, card_info):
    card_pieces = list()
    
    card_pieces.append(CARD_BEGIN)
    
    if not 'text' in card_info:
        card_text = ""
    else:
        card_text = card_info['text']
        card_text = card_text.lower() # lowercase text
        card_text = re.sub(card_name.lower(), CARDNAME, card_text) # re-sub in special token 
        card_text = re.sub("\n", f" {LF} ", card_text) # replace line feed
        card_text = re.sub("—", f" {EM} ", card_text) # replace em dash
        card_text = re.sub("\(.+\)", "", card_text) # replacce reminder text
        card_text = re.sub(":", " :", card_text) # space out activation symbol
        card_text = re.sub("}{", "} {", card_text) # space out costs
        card_text = re.sub("\.|,|\"|'", "", card_text) # remove punctuation

    card_pieces.append(card_text)
    
    card_pieces.append(CARD_END + "\n")
    
    return " ".join(card_pieces)

# generate card documents
def gen_card_docs(card_json_path):
    card_data = load_card_data(card_json_path)
    card_docs = list()

    for card_name, card_printings in card_data.items():
        # only consider latest printing
        card_info = card_printings[-1]

        if not passes_restrictions(card_info):
            continue
        card_docs.append(card_info_to_card_doc(card_name, card_info))
    return card_docs

# write card_docs to file
def gen_dataset(card_json_path, train_path, val_path, n_cards=1_000_000):
    card_docs = gen_card_docs(card_json_path)[:n_cards]
    card_train, card_val = train_test_split(card_docs, test_size=0.2)
    
    def write(path, cards):
        with open(path, "w", encoding="utf-8") as output_file:
            output_file.writelines(cards)
            output_file.close()

    write(train_path, card_train)
    write(val_path, card_val)
    

if __name__ == "__main__":
    gen_dataset("./AtomicCards.json", "./dataset/cards_train.txt", "./dataset/cards_val.txt")