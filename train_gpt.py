from transformers import GPT2LMHeadModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from load_tokenizer import load_tokenizer
from card_dataset import CardDataset
from constants import *

model_name = "gpt-mtg"

tokenizer = load_tokenizer(gpt=True)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=128,
    n_layer=8,
    n_head=8
)

model = GPT2LMHeadModel(config)

print(f"Num Parameters: {model.num_parameters()}")

training_args = TrainingArguments(
    output_dir=f"./saved/{model_name}",
    overwrite_output_dir=True,
    num_train_epochs=500,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASK_PROB)

card_trainset = CardDataset('./dataset/cards_train.txt', tokenizer, to_tensor=True)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=card_trainset
)

tokenizer.save_pretrained(f"./saved/{model_name}")
trainer.train()
trainer.save_model(f"./saved/{model_name}")