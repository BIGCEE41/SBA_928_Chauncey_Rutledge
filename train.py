"""
SBA 928 - Fine-tuning script: market-research prompt engineering.
Student: Chaucey rutledge

Fine-tunes google/flan-t5-small with LoRA for customer-support
market-research prompt tasks.

Reproduces the training run documented in
artifacts/training_evidence.json:
  model: google/flan-t5-small | method: LoRA seq2seq
  device: CPU | epochs: 1 | batch size: 2 | lr: 2e-4 | seed: 928
  train examples: 144 | validation examples: 36
  mean training loss: 2.025007

Usage:
  python train.py
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ------------------------------------------------------------------
# Config (matches training_evidence.json)
# ------------------------------------------------------------------
MODEL_ID = "google/flan-t5-small"
SEED = 928
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 128

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------
# Load training / validation data
# ------------------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


train_path = ARTIFACTS / "training_data.jsonl"
val_path = ARTIFACTS / "validation_data.jsonl"
if not train_path.exists():
    # fall back to the data/ folder copies if present
    train_path = PROJECT_ROOT / "data" / "training_data.jsonl"
    val_path = PROJECT_ROOT / "data" / "validation_data.jsonl"

train_records = load_jsonl(train_path)
val_records = load_jsonl(val_path)
print(f"Loaded {len(train_records)} training, {len(val_records)} validation examples.")

# ------------------------------------------------------------------
# Tokenizer + model with LoRA
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------------------------------------------
# Tokenize
# ------------------------------------------------------------------
def preprocess(batch):
    inputs = tokenizer(
        batch["input"], max_length=MAX_SOURCE_LEN,
        truncation=True, padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"], max_length=MAX_TARGET_LEN,
            truncation=True, padding=False,
        )
    inputs["labels"] = labels["input_ids"]
    return inputs


train_ds = Dataset.from_list(train_records).map(
    preprocess, batched=True, remove_columns=["input", "target"])
val_ds = Dataset.from_list(val_records).map(
    preprocess, batched=True, remove_columns=["input", "target"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
args = TrainingArguments(
    output_dir=str(ARTIFACTS / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    use_cpu=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

train_result = trainer.train()
mean_loss = train_result.training_loss
print(f"Mean training loss: {mean_loss:.6f}")

# ------------------------------------------------------------------
# Save adapter + training evidence
# ------------------------------------------------------------------
adapter_dir = ARTIFACTS / "fine_tuned_adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Saved LoRA adapter to {adapter_dir}")

evidence = {
    "model": MODEL_ID,
    "method": "LoRA sequence-to-sequence fine-tuning",
    "device": "CPU",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "train_examples": len(train_records),
    "validation_examples": len(val_records),
    "mean_training_loss": round(float(mean_loss), 6),
    "lora_config": {
        "r": 8, "lora_alpha": 32,
        "lora_dropout": 0.1, "target_modules": ["q", "v"],
    },
}
with open(ARTIFACTS / "training_evidence.json", "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
print("Wrote artifacts/training_evidence.json")
print("Done.")
