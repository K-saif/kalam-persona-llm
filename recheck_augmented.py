

import re

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import torch
from tqdm import tqdm

# Load data
with open("data/test_augmentaion_answer_1_word.json", "r") as f:
    data = json.load(f)


# Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)

# def no_of_question(char_count):
#     if char_count< 500:
#     	no_of_ques = 4
#     	return no_of_ques

#     if char_count >= 500 and char_count< 600:
#     	no_of_ques = 5
#     	return no_of_ques






import re

true_count = 0
false_count = 0



# Run QA generation
results = []

for i, item in tqdm(enumerate(data)):
    # char_count = len(paragraph.replace(" ", ""))
    # no_of_ques = no_of_question(char_count)

	# for augmentaion
	messages = [
	    {"role": "system", "content": "You are an AI trained to help improve question-answering datasets and check the quality of datasets. Given a question and its answer, the question is asked by a normal person and the answer is given by Dr. APJ Abdul Kalam. You must check if the question is appropriate to ask Dr. Kalam. If the question is vague, off-topic, meaningless, or not something someone would reasonably ask Dr. Kalam, then respond with true. Otherwise, respond with false. Only reply with true or false and nothing else."},
	    # {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 3 new question that asks for the same information in a different way without changing the meaning. Also, rephrase the answer appropriately. only response in the following format ,\nnew-question:\nnew-answer:"},
	    {"role": "user", "content": f'question:{item["question"]}\n answer:{item["answer"]}'},
	]
	prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
	terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
	outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
	print(outputs[0]["generated_text"][len(prompt):])
	generated_text = outputs[0]["generated_text"][len(prompt):].strip()
	match = re.search(r"\b(true|false)\b", generated_text, re.IGNORECASE)
	if match:
	    result = match.group(1).lower()
	    if result == "true":
	        true_count += 1
	        results.append(item["question"])
	        print(f"[TRUE] Question: {item['question']}")
	    else:
	        false_count += 1
	else:
	    print("[WARN] No valid true/false detected in response.")
	    continue


# qa_pairs = []
# qa_blocks = re.findall(r"new-question:\s*(.*?)\nnew-answer:\s*(.*?)(?=\nnew-question:|\Z)", generated_text.strip(), re.DOTALL)

	# # Updated regex for numbered questions and answers
	# # qa_blocks = re.findall(r"\d+\.\s*(.*?)\nAnswer:\s*(.*?)(?=\n\d+\.|\Z)", generated_text.strip(), re.DOTALL)

	# # Include the original QA pair first
	# qa_pairs.append({"question": item["question"],"answer": item["answer"]})
	# # qa_pairs.append({"context": item["context"],"question": item["question"],"answer": item["answer"]})

	# # qa_blocks = re.findall(
	# #     r"new[- ]question[- ]\d+:\s*(.*?)\nnew[- ]answer(?:[- ]\d*)?:\s*(.*?)(?=\nnew[- ]question[- ]\d+:|\Z)",
	# #     generated_text.strip(),
	# #     re.DOTALL
	# # )


	# for q, a in qa_blocks:
	# 	qa_pairs.append({
	# 		# "context": item["context"],
	# 		"question": q.strip(),
	# 		"answer": a.strip()
	# 		})

	# results.extend(qa_pairs)

	# print("Extracted QA blocks:", qa_blocks)

	print(f"✅ Processed paragraph {i+1}")


# with open("data/test_augmentaion_answer_1_word.json", "w") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)


print("🎉 All Q&A pairs saved ")



print(results)
print("this is true: ",true_count)
print("this is false: ",false_count)
