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

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                                      # Slightly higher for SFT
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load SFT dataset
dataset = load_dataset("json", data_files="kalam_sft_final.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field=None,           # Important when using "messages"
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,                     # False for chat format
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,     # Adjust based on VRAM (try 8-16)
        num_train_epochs=2,                 # 2 epochs is good
        learning_rate=8e-5,                 # Lower than CPT
        warmup_steps=20,
        logging_steps=10,
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