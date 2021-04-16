from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer 
from card_dataset import CardDataset
from constants import *

from transformers import OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

model_name = "gpt1"

tokenizer = OpenAIGPTTokenizer(
    './tokenizer/vocab.json', './tokenizer/merges.txt',
    bos_token=CARD_BEGIN, eos_token=CARD_END, sep_token=CARD_END, 
    unk_token=UNK, pad_token=CARD_PAD, mask_token=CARD_MASK
)

config = OpenAIGPTConfig(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=64,
    n_layer=4,
    n_head=8
)

model = OpenAIGPTLMHeadModel(config)

print(f"Num Parameters: {model.num_parameters()}")

training_args = TrainingArguments(
    output_dir=f"./saved/{model_name}",
    overwrite_output_dir=True,
    num_train_epochs=400,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASK_PROB)

card_trainset = CardDataset('./dataset/cards_train.txt', tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=card_trainset
)

tokenizer.save_vocabulary(f"./saved/{model_name}")
trainer.train()
trainer.save_model(f"./saved/{model_name}")