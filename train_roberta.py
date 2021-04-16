from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer 
from card_dataset import CardDataset
from constants import *

model_name = "transformer0"

tokenizer = RobertaTokenizer(
    './tokenizer/vocab.json', './tokenizer/merges.txt',
    bos_token=CARD_BEGIN, eos_token=CARD_END, sep_token=CARD_END, 
    unk_token=UNK, pad_token=CARD_PAD, mask_token=CARD_MASK,
)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=1
)

model = RobertaForMaskedLM(config=config)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASK_PROB)

training_args = TrainingArguments(
    output_dir=f"./models/{model_name}",
    overwrite_output_dir=False,
    num_train_epochs=80,
    per_device_train_batch_size=128,
    save_steps=200,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_steps=100,
    do_train=True,
    fp16=True
)

card_trainset = CardDataset('./dataset/cards_train.txt', tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=card_trainset
)

tokenizer.save_vocabulary(f"./models/{model_name}")
trainer.train()