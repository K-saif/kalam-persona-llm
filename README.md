# APJ Abdul Kalam Persona LLM

This project fine-tunes Large Language Models (LLMs) to emulate the inspirational communication style and philosophical thinking of Dr. APJ Abdul Kalam.

The training pipeline uses:
- Continued Pre-Training (CPT)
- LoRA-based Supervised Fine-Tuning (SFT)
- QLoRA / 4-bit optimization for consumer GPUs

Pipeline:
Qwen2.5-7B → CPT → Merge → SFT → Kalam Persona LoRA

## 📋 Project Overview

The goal is to create a persona-adapted LLM that can:
- Generate responses inspired by the communication style, wisdom, and philosophy of APJ Abdul Kalam
- Answer questions based on his famous speeches and writings
- Provide inspirational insights from his perspective
- Respond in a Kalam-inspired conversational style

## 🚀 Quick Start

### Inference Pretrained Model

This repository contains LoRA adapter weights.

Base model required:
`Qwen/Qwen2.5-7B`

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B"
adapter = "K-saif/apj-kalam-instruct-v2"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter)

messages = [
    {
        "role": "user",
        "content": "What is the purpose of life?"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    repetition_penalty=1.1,
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print(response)
```

### Train Your Own Model

**Resources:**
- 🤗 **Model**: [K-saif/apj-kalam-instruct-v2](https://huggingface.co/K-saif/apj-kalam-instruct-v2)
- 📊 **Dataset**: [K-saif/apj-kalam-instruct-dataset-v2](https://huggingface.co/datasets/K-saif/apj-kalam-instruct-dataset-v2)

#### Prerequisites

```bash
pip install torch transformers datasets trl peft
```

## 🧠 Training Pipeline

The model was trained using a multi-stage fine-tuning pipeline:

```text
Qwen2.5-7B Base Model
        ↓
Continued Pretraining (CPT)
        ↓
Merge CPT Weights
        ↓
Supervised Fine-Tuning (SFT)
        ↓
Final APJ Abdul Kalam Persona LoRA
```

This approach improves:
- domain adaptation
- conversational quality
- personality consistency
- philosophical response generation

## 📈 Training Summary

- Base Model: Qwen2.5-7B
- Training Method: CPT + SFT
- Quantization: QLoRA (4-bit)
- Final SFT Eval Accuracy: ~83%


## 💻 Hardware & Optimization

The project uses:
- QLoRA / 4-bit quantization
- PEFT (Parameter-Efficient Fine-Tuning)
- Gradient checkpointing
- Consumer GPU-friendly training setup


## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.
