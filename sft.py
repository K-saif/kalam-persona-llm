import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from trl import SFTTrainer, SFTConfig

# ========================= CONFIG =========================

model_name = "kalam_cpt_merged"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

# ========================= TOKENIZER =========================

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========================= MODEL =========================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

# IMPORTANT
model = prepare_model_for_kbit_training(model)

# ========================= NEW SFT LORA =========================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# ========================= DATASET =========================

dataset = load_dataset(
    "json",
    data_files="kalam_sft_final_fixed.jsonl",
    split="train"
)

# ========================= FORMAT DATASET =========================

def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_example)

# OPTIONAL BUT STRONGLY RECOMMENDED
dataset = dataset.train_test_split(
    test_size=0.03,
    seed=3407
)

# ========================= TRAINING CONFIG =========================

sft_config = SFTConfig(
    output_dir="kalam_sft_qwen7b",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=2e-5,

    num_train_epochs=3,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    optim="paged_adamw_8bit",

    logging_steps=20,

    save_strategy="epoch",

    eval_strategy="steps",
    eval_steps=100,

    bf16=True,

    gradient_checkpointing=True,

    max_grad_norm=0.3,

    weight_decay=0.01,

    report_to="none",

    seed=3407,

    remove_unused_columns=False,

    packing=False,

    max_length=1024,
)

# ========================= TRAINER =========================

trainer = SFTTrainer(
    model=model,

    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],

    args=sft_config,

    processing_class=tokenizer,
)

# ========================= TRAIN =========================

print("🚀 Starting Proper SFT Training...")

trainer.train()

# ========================= SAVE =========================

trainer.model.save_pretrained("kalam_final_lora")
tokenizer.save_pretrained("kalam_final_lora")

print("✅ SFT Training Completed!")