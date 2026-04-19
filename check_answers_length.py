import os
import json
from collections import defaultdict

# Adjust this to the path where your JSON files are stored
folder_path = "/home/oem/Music/temp1/data/All_Merged/test_7_may/eval"

# Define the word length ranges
buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30),(31, 35), (36, 40), (41, 50)]
bucket_counts = defaultdict(int)
overflow_count = 0

def get_bucket(word_count):
    for low, high in buckets:
        if low <= word_count <= high:
            return f"{low}-{high}"
    return f"{high+1}+"
i=0
# Loop through all JSON files
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        print(filename)
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):  # In case each file contains a list of QA
                for item in data:
                    answer = item.get("answer", "")
                    word_count = len(answer.strip().split())
                    bucket = get_bucket(word_count)
                    bucket_counts[bucket] += 1
                    i=i+1
            else:  # In case file has a single QA object
                # print("hereeeeee")
                answer = data.get("answer", "")
                word_count = len(answer.strip().split())
                bucket = get_bucket(word_count)
                bucket_counts[bucket] += 1

# Sort and print the results
for bucket in sorted(bucket_counts, key=lambda x: int(x.split("-")[0]) if "-" in x else 1000):
    print(f"Answers with {bucket} words: {bucket_counts[bucket]}")

print(i)