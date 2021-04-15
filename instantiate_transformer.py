from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from constants import *

tokenizer = RobertaTokenizerFast(
    './tokenizer/vocab.json', './tokenizer/merges.txt'
)

config = RobertaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=1024,
    num_attention_heads=3,
    num_hidden_layers=1
)

model = RobertaForMaskedLM(config=config)

print(f"Num parameters: {model.num_parameters()}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASK_PROB)

training_args = TrainingArguments(
    output_dir="./models/transformer0",
    overwrite_output_dir=False,
    num_train_epochs=1,
    per_device_train_batch_size=12,
    save_steps=200,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator
)

#trainer.save_model('./models/transformer0')