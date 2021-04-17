from load_tokenizer import load_tokenizer
from keras.models import load_model
import numpy as np
from constants import *

class LSTMGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = load_tokenizer()
        self.model = load_model(f"./saved/{model_name}")

        self.seq_len = self.model.layers[0].input_shape[1]
        self.tokenizer.enable_padding(direction="left", pad_token=CARD_PAD, length=self.seq_len)

    def generate(self, prompt, use_sampling=False):
        encoded = self.tokenizer.encode(prompt)
        x = np.expand_dims(np.array(encoded.ids[-self.seq_len:]), axis=0)
        prediction = self.model.predict(x).flatten()

        if not use_sampling:
            return self.tokenizer.id_to_token(np.argmax(prediction))
        else:
            sampled_id = np.random.choice(np.arange(self.tokenizer.get_vocab_size()), p=prediction)
            return self.tokenizer.id_to_token(sampled_id)

if __name__ == "__main__":
    model_name = "lstm-mtg"
    generator = LSTMGenerator(model_name)

    prompt = "<s> when CARDNAME enters the battlefield"
    print(f"Input prompt: \"{prompt}\"")

    prompt += " " + generator.generate(prompt)
    print(f"Output prompt: \"{prompt}\"")
