# def normalize_paragraphs(input_file, output_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     paragraphs = []
#     current_para = []

#     for line in lines:
#         stripped = line.strip()

#         if stripped == "":
#             if current_para:
#                 # Join paragraph into single line
#                 paragraph = " ".join(current_para)
#                 paragraphs.append(paragraph)
#                 current_para = []
#         else:
#             current_para.append(stripped)

#     # Add last paragraph if exists
#     if current_para:
#         paragraph = " ".join(current_para)
#         paragraphs.append(paragraph)

#     # Write to output file (each paragraph = one line)
#     with open(output_file, "w", encoding="utf-8") as f:
#         for para in paragraphs:
#             f.write(para + "\n")


    
# # Example usage
# input_file = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\wings_of_fire.txt"
# output_file = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\wings_of_fire_converted.txt"

# normalize_paragraphs(input_file, output_file)


import json
import re

INPUT_FILE = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\qa_wings_of_fire.json"
OUTPUT_FILE = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\qa_wings_of_fire1.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all JSON arrays in the file
    blocks = re.findall(r"\[\s*{.*?}\s*\]", content, re.DOTALL)

    all_data = []

    for block in blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                all_data.extend(parsed)
        except Exception as e:
            print("Skipping invalid block:", e)

    # Save merged output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(all_data)} QA pairs into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()