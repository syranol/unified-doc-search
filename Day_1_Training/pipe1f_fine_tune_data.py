"""Utility helpers for fine-tuning experiments from the Day 1 training notebook."""

from typing import Mapping

from transformers import Trainer, TrainingArguments


def run_training(
    model,
    tokenized_datasets: Mapping[str, object],
    data_collator,
    tokenizer,
    output_dir: str = "test-trainer",
):
    """Configure and run the HuggingFace ``Trainer`` with pre-built artifacts."""

    training_args = TrainingArguments(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer


if __name__ == "__main__":
    raise SystemExit(
        "This module exposes `run_training`; provide model/tokenizer artifacts from the "
        "preceding setup steps before calling it."
    )
