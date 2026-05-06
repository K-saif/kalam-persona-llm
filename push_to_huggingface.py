from unsloth import FastLanguageModel
from datasets import load_dataset

# =====================================================
# CONFIG FLAGS
# =====================================================

UPLOAD_MODEL = False
UPLOAD_DATASET = True

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_PATH = "C:/Users/khans/Documents/APJ-Abdul-Kalam/kalam_final_model"
MODEL_REPO = "K-saif/apj-kalam-instruct"

MAX_SEQ_LENGTH = 1024

# =====================================================
# DATASET CONFIG
# =====================================================

DATASET_FILE = "C:/Users/khans/Documents/APJ-Abdul-Kalam/Datasets/kalam_sft_final_fixed.jsonl"
DATASET_REPO = "K-saif/apj-kalam-instruct-dataset"

# =====================================================
# UPLOAD MODEL
# =====================================================

if UPLOAD_MODEL:

    print("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    print("Uploading model to Hugging Face...")

    model.push_to_hub(MODEL_REPO)
    tokenizer.push_to_hub(MODEL_REPO)

    print(f"✅ Model uploaded successfully -> {MODEL_REPO}")

# =====================================================
# UPLOAD DATASET
# =====================================================

if UPLOAD_DATASET:

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=DATASET_FILE,
        split="train"
    )

    print("Uploading dataset to Hugging Face...")

    dataset.push_to_hub(DATASET_REPO)

    print(f"✅ Dataset uploaded successfully -> {DATASET_REPO}")

# =====================================================
# DONE
# =====================================================

print("🎉 Upload process completed!")