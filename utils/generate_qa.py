import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# CONFIG
# =========================
MODEL_NAME = "google/gemma-3b-it"
INPUT_FILE = "path/to/input.txt"
OUTPUT_FILE = "path/to/output.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL (4-bit optional)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# =========================
# PROMPT TEMPLATE
# =========================
BASE_PROMPT = """You are generating high-quality question-answer pairs for fine-tuning a language model to mimic the speaking and thinking style of Dr. A. P. J. Abdul Kalam, based on his book "Wings of Fire".

You will be given a paragraph from the book.

Your task:
- Generate 5 to 10 meaningful question-answer pairs based ONLY on the given paragraph.
- The questions should feel like someone is genuinely asking Dr. Kalam about his life, experiences, thoughts, or lessons.
- The answers should be written in the FIRST PERSON, as if Dr. Kalam himself is responding.

Style Guidelines for Answers:
- Keep answers VERY CONCISE (STRICT: 1 sentence, max 25–35 words)
- Reflect humility and clarity of thought
- Include light reflection, but avoid long philosophical explanations
- Use simple and natural language
- Stay grounded in the paragraph content

Style Guidelines for Questions:
- Natural, conversational, and curious tone
- Focus on "why", "how", "what did you feel", "what did you learn"
- Avoid generic or robotic questions

Output Format (STRICT JSON):
[
  {
    "question": "....",
    "answer": "...."
  }
]

Important:
- Each answer MUST be only ONE sentence (no multi-line answers)
- Do NOT exceed 35 words per answer
- Do NOT copy sentences directly from the paragraph
- Do NOT hallucinate events outside the paragraph
- Ensure answers still feel like Kalam’s voice: simple, wise, and grounded
- Maintain diversity in questions

Now here is the paragraph:
"""

# =========================
# GENERATE FUNCTION
# =========================
def generate_qa(paragraph):
    prompt = BASE_PROMPT + paragraph.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only JSON part
    start = text.find("[")
    end = text.rfind("]") + 1

    if start != -1 and end != -1:
        json_text = text[start:end]
        try:
            return json.loads(json_text)
        except:
            print("⚠️ JSON parsing failed, skipping...")
            return []
    else:
        print("⚠️ No JSON found, skipping...")
        return []

# =========================
# MAIN PIPELINE
# =========================
def main():
    # Load existing output if exists
    try:
        with open(OUTPUT_FILE, "r") as f:
            all_data = json.load(f)
    except:
        all_data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        paragraphs = f.readlines()

    for para in tqdm(paragraphs):
        if not para.strip():
            continue

        qa_pairs = generate_qa(para)

        if isinstance(qa_pairs, list):
            all_data.extend(qa_pairs)

    # Save final JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_data)} QA pairs to {OUTPUT_FILE}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()