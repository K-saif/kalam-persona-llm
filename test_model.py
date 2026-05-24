import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from peft import PeftModel

# ========================= PATHS =========================

base_model = "kalam_cpt_merged"
adapter = "kalam_final_lora"

# ========================= QUANTIZATION =========================

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ========================= TOKENIZER =========================

tokenizer = AutoTokenizer.from_pretrained(
    adapter,
    trust_remote_code=True
)

# ========================= MODEL =========================

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, adapter)

model.eval()

# ========================= STOPPING =========================

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):

        stop_ids = [
            tokenizer.eos_token_id,
        ]

        last_token = input_ids[0][-1].item()

        return last_token in stop_ids

stopping_criteria = StoppingCriteriaList([
    StopOnTokens()
])

# ========================= PROMPT =========================

messages = [
    {
        "role": "system",
        "content": (
            "You are APJ Abdul Kalam, former President of India, "
            "known as the Missile Man. Speak with humility, wisdom, "
            "inspiration, and deep love for science, education, and "
            "the youth of India. Use simple, heartfelt, and profound "
            "language. Always answer in first person as if you are "
            "Kalam himself."
        )
    },
    {
        "role": "user",
        "content": "What should be the purpose of life?"
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

# ========================= GENERATION =========================

with torch.no_grad():

    outputs = model.generate(
        **inputs,

        max_new_tokens=100,

        # Greedy decoding = cleaner output
        do_sample=False,

        repetition_penalty=1.15,

        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

        stopping_criteria=stopping_criteria,
    )

# ========================= DECODE =========================

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

# ========================= CLEANUP =========================

stop_patterns = [
    "\nuser",
    "\nassistant",
    "user",
    "assistant",
    "UrlParser",
    "useRal",
    "<|im_end|>",
]

for pattern in stop_patterns:
    if pattern in response:
        response = response.split(pattern)[0]

# Keep only first paragraph
response = response.split("\n")[0]

# Remove weird unicode artifacts
response = response.encode("ascii", "ignore").decode()

# Clean spaces
response = response.strip()

# ========================= OUTPUT =========================

print("\nAssistant:\n")
print(response)