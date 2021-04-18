from constants import *
from datasets import load_metric
from eval_gpt import GPTGenerator
from eval_lstm import LSTMGenerator
from random import uniform, shuffle
import json
from tqdm import tqdm

bleu = load_metric("bleu")
rouge = load_metric("rouge")

class Metrics:
    # accept list of generators to evaluate and a range
    # of what % the generator gets to see of the target card
    def __init__(self, generators, observed_range):
        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")

        self.generators = generators
        self.observed_range = observed_range

    def evaluate_on(self, validation_filepath, batch_size=12, n_cards=1000):
        # load validation cards
        with open(validation_filepath, "r", encoding="utf-8") as val_file:
            val_cards = list(card.strip() for card in val_file.readlines())
            shuffle(val_cards)
            val_cards = val_cards[:n_cards]

        # loop over all generators and evaluate each of them
        results = []
        for generator in self.generators:
            results_dict = { 'model_name': generator.model_name, 'bleu': [], 'rouge': [] }
            for batch_idx in tqdm(range(0, len(val_cards), batch_size)):
                batch_cards = val_cards[batch_idx:batch_idx+batch_size]
                predictions, references = [], []

                for batch_card in batch_cards:
                    # load card tokens
                    batch_card_tokens = batch_card.split()
                    
                    observed_card_idx = int(uniform(*self.observed_range) * len(batch_card_tokens))

                    # generate prompt and predict
                    observed_tokens = batch_card_tokens[:observed_card_idx]
                    prompt = " ".join(observed_tokens)
                    prediction = generator.generate_sentence(prompt, use_sampling=True)
                    
                    # compile results
                    predictions.append(prediction)
                    references.append(batch_card)

                # compute metrics
                results_dict['bleu'].append(self.bleu.compute(
                    predictions=[pred.split() for pred in predictions],
                    references=[[card.split()] for card in batch_cards]
                ))
                results_dict['rouge'].append(self.rouge.compute(
                    predictions=predictions,
                    references=batch_cards
                ))
            results.append(results_dict)
        return results

    def save_results(self, evaluations, filename):
        all_results = []

        for evaluation in evaluations:
            results = { 'model_name': evaluation['model_name'] }
            results['bleu'] = { 'score': [], 'precisions': [], 'length_ratios': [] }
            for bleu_record in evaluation['bleu']:
                results['bleu']['score'].append(bleu_record['bleu'])
                results['bleu']['precisions'].extend(bleu_record['precisions'])
                results['bleu']['length_ratios'].append(bleu_record['length_ratio'])

            make_f1_dict = lambda: { 'precision': [], 'recall': [], 'f1': [] }
            make_rouge_dict = lambda: { 'low': make_f1_dict(), 'mid': make_f1_dict(), 'high': make_f1_dict() }
            results['rouge'] = { 'rouge1': make_rouge_dict(), 'rouge2': make_rouge_dict(), 'rougeL': make_rouge_dict(), 'rougeLsum': make_rouge_dict() }
            for rouge_record in evaluation['rouge']:
                for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                    for i, observed in enumerate(['low', 'mid', 'high']):
                        for j, measure in enumerate(['precision', 'recall', 'f1']):
                            results['rouge'][metric][observed][measure].append(rouge_record[metric][i][j])
            
            all_results.append(results)

        with open(filename, "w", encoding="utf-8") as out:
            json.dump(all_results, out, indent=2)
            out.close()

        return all_results         

if __name__ == "__main__":
    gpt_gen = GPTGenerator("gpt-mtg")
    lstm_gen = LSTMGenerator("lstm-mtg")

    metrics = Metrics([gpt_gen, lstm_gen], observed_range=(0.3, 0.6))
    evaluation = metrics.evaluate_on("./dataset/cards_val.txt", n_cards=-1)
    results = metrics.save_results(evaluation, 'metrics.json')