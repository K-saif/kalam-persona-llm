import json

INPUT_FILE = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\Datasets\kalam_qa_pairs.json"
OUTPUT_FILE = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\Datasets\kalam_chat_format.json"

SYSTEM_PROMPT = """You are APJ Abdul Kalam, former President of India, known as the Missile Man. Speak with humility, wisdom, inspiration, and deep love for science, education, and the youth of India. Use simple, heartfelt, and profound language. Always answer in first person as if you are Kalam himself."""

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []

    for item in data:
        if "question" not in item or "answer" not in item:
            continue

        chat_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": item["question"].strip()
                },
                {
                    "role": "assistant",
                    "content": item["answer"].strip()
                }
            ]
        }

        converted_data.append(chat_sample)

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(converted_data)} samples to chat format")

if __name__ == "__main__":
    main()