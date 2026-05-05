import json

# Read your current file (which is a big array)
with open(r"C:\Users\khans\Documents\APJ-Abdul-Kalam\Datasets\kalam_chat_format.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)          # Load the big array

# Write it as proper JSON Lines
with open(r"C:\Users\khans\Documents\APJ-Abdul-Kalam\Datasets\kalam_sft_final_fixed.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Fixed! Converted {len(data)} examples to proper JSONL format.")