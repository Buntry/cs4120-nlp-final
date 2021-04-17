from constants import *
from datasets import load_metric
from eval_gpt import GPTGenerator
from eval_lstm import LSTMGenerator

bleu = load_metric("bleu")
rouge = load_metric("rouge")

class Metrics:
    def __init__(self, generators):
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")

        self.generators = generators 

if __name__ == "__main__":


    gpt_gen = GPTGenerator()

    metrics = Metrics()