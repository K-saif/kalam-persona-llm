import json
import os

INPUT_FOLDER = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\old\data\All_Merged\training_6_may_personal"   # folder containing all JSON files
OUTPUT_FILE = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\Datasets\GK\merged_qa.json"

def main():
    all_data = []

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".json"):
            file_path = os.path.join(INPUT_FOLDER, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        print(f"⚠️ Skipping (not a list): {filename}")

            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")

    # Optional: remove duplicates
    unique = {(d["question"], d["answer"]): d for d in all_data if "question" in d and "answer" in d}
    all_data = list(unique.values())

    # Save merged file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(all_data)} unique QA pairs into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()