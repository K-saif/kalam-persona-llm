from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from huggingface_hub import login
login(token="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", add_to_git_credential=False)

model_id = "meta-llama/Meta-Llama-3-8B"  # or another variant

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # use float16 if on GPU
    device_map="auto"            # auto places model across GPUs if needed
)


# model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
# rsult= pipeline("Hey how are you doing today?")
# print(rsult)


while True:

    # Inference test
    prompt = input("\n\nquestion:")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))






# # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# # pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3", trust_remote_code=True)
# pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3", trust_remote_code=True, device=0)  # 0 for first GPU
# pipe(messages)



# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_id = "deepseek-ai/deepseek-llm-7b-base"  # Or deepseek-llm-7b-chat

# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype="auto",
#     trust_remote_code=True
# )

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# response = pipe("Who are you?", max_new_tokens=100)
# print(response[0]['generated_text'])
