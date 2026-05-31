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
    device_map="auto",
)

model = PeftModel.from_pretrained(model, adapter)

model.eval()

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

with torch.no_grad():

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

if "<END>" in response:
    response = response.split("<END>")[0]

print(response)
