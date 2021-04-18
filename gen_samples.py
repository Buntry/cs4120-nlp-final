from eval_gpt import GPTGenerator
from eval_lstm import LSTMGenerator
import json
import random

# Real card generator, used to generate real cards
class RealCardTextGenerator:
    def __init__(self, card_path):
        self.model_name = "real_cards"
        with open(card_path, "r", encoding="utf-8") as card_file:
            self.cards = list(card_file.readlines())
    
    def generate(self, prompt, use_sampling=False):
        assert False, "Cannot generate one token with CardTextGenerator"
    
    def generate_sentence(self, prompt, use_sampling=False):
        idx = random.randrange(len(self.cards))
        return self.cards[idx]

def generate_samples(generators, n_cards=10):
    samples = []
    for generator in generators:
        for _ in range(n_cards):
            samples.append({
                'name': generator.model_name, 
                'card': generator.generate_sentence("", use_sampling=True).strip()
            })
    return samples

if __name__ == "__main__":
    gpt_gen = GPTGenerator("gpt-mtg")
    lstm_gen = LSTMGenerator("lstm-mtg")
    real_cards = RealCardTextGenerator("./dataset/cards_val.txt")

    samples = generate_samples([gpt_gen, lstm_gen, real_cards], n_cards=10)
    with open("samples.json", "w", encoding="utf-8") as out:
        json.dump(samples, out, indent=2)