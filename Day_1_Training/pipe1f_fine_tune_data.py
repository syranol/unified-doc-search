# 1. Prepare dataset
# 2. Load pretrained Tokenizer, call it with dataset -> encoding
# 3. Build PyTorch Dataset with encodings
# 4. Load pretrained Models
# 5. a) Load Trainer and train int
#    b) Native pyTorch training loop

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["trains"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()