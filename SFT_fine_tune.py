import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ========================= CONFIG =========================
max_seq_length = 1024

# Load your CPT model as base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="kalam_cpt_lora_v2",        # ← Change if you have final version
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Formatting function for chat data
def formatting_func(example):
    outputs = []

    # Detect if batched or single
    if isinstance(example["messages"], list) and len(example["messages"]) > 0 and isinstance(example["messages"][0], dict):
        # 👉 SINGLE sample case
        batch = [example["messages"]]
    else:
        # 👉 BATCHED case
        batch = example["messages"]

    for messages in batch:
        text = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue  # safety check

            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"

        outputs.append(text)

    return outputs
# Load SFT dataset
dataset = load_dataset("json", data_files="C:/Users/khans/Documents/APJ-Abdul-Kalam/Datasets/kalam_sft_final_fixed.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,                     # False for chat format
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,     # Adjust based on VRAM (try 8-16)
        num_train_epochs=2,                 # 2 epochs is good
        learning_rate=8e-5,                 # Lower than CPT
        warmup_steps=20,
        logging_steps=5,
        output_dir="kalam_sft_output",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    ),
)

print("Starting Supervised Fine-Tuning (SFT)...")
trainer.train()

model.save_pretrained("kalam_final_model")
tokenizer.save_pretrained("kalam_final_model")