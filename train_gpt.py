from transformers import GPT2LMHeadModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from load_tokenizer import load_tokenizer
from card_dataset import CardDataset
from constants import *

# Trainer for GPT2

class GPT2Trainer:
    def __init__(self, model_name, train_path, n_positions=512, n_ctx=512, n_embd=128, n_layer=8, n_head=8):
        self.model_name = model_name
        self.tokenizer = load_tokenizer(gpt=True)
        self.train_path = train_path

        self.config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=n_positions,
            n_ctx=n_ctx,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head
        )

        self.model = GPT2LMHeadModel(self.config)

    # trains and saves a gpt2 trainer
    def train(self, num_epochs=500, batch_size=32, save_total_limit=2, save_steps=500, logging_steps=100):
        training_args = TrainingArguments(
            output_dir=f"./saved/{self.model_name}",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            prediction_loss_only=True,
            logging_steps=logging_steps
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=MASK_PROB)
        card_trainset = CardDataset(self.train_path, self.tokenizer, to_tensor=True)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=card_trainset
        )

        self.tokenizer.save_pretrained(f"./saved/{self.model_name}")
        trainer.train()
        trainer.save_model(f"./saved/{self.model_name}")

if __name__ == "__main__":
    trainer = GPT2Trainer("gpt-mtg", "./dataset/cards_train.txt")
    trainer.train()