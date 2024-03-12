import os
import sys
from transformers import RobertaForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from nlp.test_data import sample_data
import numpy as np
import evaluate

# Construct the absolute path to trained_model.pth in the project root
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pth")

# Instantiate a new RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
metric = evaluate.load("accuracy")  # Fix typo in accuracy

# Load and convert from list of dictionaries to a DataFrame
import pandas as pd
df = pd.DataFrame(sample_data)

# Extract features and labels from the DataFrame
features = df["text"].tolist()
labels = df["label"].tolist()

# Split data
train_features, eval_features, train_labels, eval_labels = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

# Tokenize and preprocess the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Tokenize the training dataset
tokenized_train_data = Dataset.from_dict({"text": train_features, "label": train_labels}).map(tokenize_function, batched=True)

# Tokenize the evaluation dataset
tokenized_eval_data = Dataset.from_dict({"text": eval_features, "label": eval_labels}).map(tokenize_function, batched=True)

# Convert logits to predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# For monitoring
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_eval_data,
    compute_metrics=compute_metrics,
)

# Training will initialize some weights randomly, and it's expected behavior.
# The warning about weights not being pretrained is informative, not an error.
# If you see actual errors during training, please provide more details.
trainer.train()

# Save the trained model
trainer.save_model(model_path)
