"""
Train a small model with some json question/response inputs
"""


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import os

MODEL_NAME = "tiiuae/falcon-rw-1b"
DATA_PATH = "sample_train.jsonl"
OUTPUT_DIR = "./lora-output"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def combine_fields(example):
    example["text"] = f"### Question:\n{example['prompt']}\n### Answer:\n{example['response']}"
    return example

dataset = dataset.map(combine_fields)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# ---- Load Model ----
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- LoRA Adapter ----
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

# ---- Training Arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
    fp16=False  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
