import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# ========================= CONFIG =========================
model_name = "Qwen/Qwen2.5-7B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====================== LOAD & TOKENIZE DATASET ======================
dataset = load_dataset("json", data_files="kalam_cpt.jsonl", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,           # We'll pad dynamically
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],     # Remove raw text after tokenization
    num_proc=4
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ====================== TRAINING =========================
training_args = TrainingArguments(
    output_dir="kalam_cpt_qwen7b",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1.5e-4,
    num_train_epochs=3,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=20,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    weight_decay=0.01,
    report_to="none",
    seed=3407,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    args=training_args,
)

print("🚀 Starting Continued Pre-Training on Qwen2.5-7B...")
trainer.train()

model.save_pretrained("kalam_cpt_lora_qwen7b")
tokenizer.save_pretrained("kalam_cpt_lora_qwen7b")
print("✅ CPT Completed!")