import re
from transformers import pipeline
from transformers import GPT2LMHeadModel
from load_tokenizer import load_tokenizer
from constants import CARD_END, CARD_PAD

class GPTGenerator:
    def __init__(self, model_name):
        self.model_name = model_name

        self.tokenizer = load_tokenizer(gpt=True, load_from=f"./saved/{self.model_name}")
        self.model = GPT2LMHeadModel.from_pretrained(
            f"./saved/{self.model_name}", 
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
    
    def generate(self, prompt, use_sampling=False):
        encoded_len = len(self.tokenizer.encode(prompt))
        prediction = self.generator(prompt, do_sample=use_sampling, return_tensors=True, max_length=encoded_len+1)
        predicted_id = prediction[0]['generated_token_ids'][encoded_len]

        return self.tokenizer.convert_ids_to_tokens(predicted_id)

    def generate_sentence(self, prompt, use_sampling=False, max_len=200):
        if not prompt:
            prompt = CARD_PAD

        prediction = self.generator(prompt, do_sample=use_sampling, max_length=max_len)[0]['generated_text']
        
        # remove padding tokens
        prediction = re.sub(CARD_PAD, "", prediction)

        # optionally slice off extra card_end items
        if CARD_END in prediction:
            return prediction[:prediction.find(CARD_END) + len(CARD_END)]
        else:
            return prediction


if __name__ == "__main__":
    model_name = "gpt-mtg"

    generator = GPTGenerator(model_name)
    prompt = "<s> when CARDNAME enters the battlefield "
    print(f"Input prompt: \"{prompt}\"")

    print(f"Output prompt: \"{generator.generate_sentence(prompt, use_sampling=True)}\"")
