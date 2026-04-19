from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# === CONFIG ===
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
peft_model_path = "/home/oem/Music/temp1/output/augmentation_eval_qa_merged_v1_llama3_QA_10000_spl_tocken/checkpoint-10000"  # Or latest/best
merged_model_output_dir = "/home/oem/Music/temp1/output/augmentation_eval_qa_merged_v1_llama3_QA_10000_spl_tocken/checkpoint-10000/merged_model"

# === LOAD BASE + MERGE ===
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto")

print("🔌 Loading PEFT (LoRA) weights...")
model = PeftModel.from_pretrained(model, peft_model_path)

print("🧠 Merging LoRA weights...")
model = model.merge_and_unload()

# === SAVE FINAL MODEL ===
print(f"💾 Saving merged model to {merged_model_output_dir}...")
model.save_pretrained(merged_model_output_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(merged_model_output_dir)

print("✅ Merged model saved successfully.")
