from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-7B"
adapter_path = "kalam_cpt_lora_qwen7b"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, adapter_path)

merged_model = model.merge_and_unload()

merged_model.save_pretrained("kalam_cpt_merged")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained("kalam_cpt_merged")