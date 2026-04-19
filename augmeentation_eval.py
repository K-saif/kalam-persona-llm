# import re
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import pipeline
# import torch
# from tqdm import tqdm

# # Load data
# with open("data/test.json", "r") as f:
#     data = json.load(f)


# # Load model and tokenizer
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# pipeline = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         "quantization_config": {"load_in_4bit": True},
#         "low_cpu_mem_usage": True,
#     },
# )


# # Run QA generation
# results = []
# eval_results = []
# for i, item in tqdm(enumerate(data)):
# 	# char_count = len(paragraph.replace(" ", ""))
# 	# no_of_ques = no_of_question(char_count)

# 	# for augmentaion
# 	messages = [
# 		#for without context
# 	    # {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 3 new answer that response for the same information in a different way without changing the meaning. write some answer in one word if it is applicable. only response in the following format ,\nnew-question:\nnew-answer:"},
	    
# 	    {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 3 new question that asks for the same information in a different way without changing the meaning. Also, rephrase the answer appropriately. only response in the following format ,\nnew-question:\nnew-answer: \nnew-question:\nnew-answer: \nnew-question:\nnew-answer:"},
# 	    {"role": "user", "content": f'question:{item["question"]}\n answer:{item["answer"]}'},
# 	]
# 	prompt = pipeline.tokenizer.apply_chat_template(
# 	    messages,
# 	    tokenize=False,
# 	    add_generation_prompt=True
# 	)
# 	terminators = [
# 	    pipeline.tokenizer.eos_token_id,
# 	    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# 	]
# 	outputs = pipeline(
# 	    prompt,
# 	    max_new_tokens=1024,
# 	    eos_token_id=terminators,
# 	    do_sample=True,
# 	    temperature=0.6,
# 	    top_p=0.9)
# 	generated_text = outputs[0]["generated_text"][len(prompt):]
# 	print("##################",generated_text)

# 	qa_pairs = []
# 	qa_blocks = re.findall(
# 	    r"new-question:\s*(.*?)\nnew-answer:\s*(.*?)(?=\nnew-question:|\Z)",
# 	    generated_text.strip(), re.DOTALL
# 	)

# 	# qa_pairs.append({"context": item["context"],"question": item["question"],"answer": item["answer"]})
# 	qa_pairs.append({"question": item["question"],"answer": item["answer"]})

# 	# Take one QA pair for eval and rest for test
# 	if qa_blocks:
# 	    q_eval, a_eval = qa_blocks[0]
# 	    eval_results.append({
# 	        "question": q_eval.strip(),
# 	        "answer": a_eval.strip()
# 	    })
# 	    for q, a in qa_blocks[1:]:
# 	    	qa_pairs.append({
# 				# "context": item["context"],
# 				"question": q.strip(),
# 				"answer": a.strip()
# 				})

# 	    results.extend(qa_pairs)

# 	print("Extracted QA blocks:", qa_blocks)
# 	print(f"✅ Processed paragraph {i+1}")

# # Save test.json
# with open("data/augmentation_eval_test.json", "w") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# # Save eval.json
# with open("data/augmentation_eval_test_eval.json", "w") as f:
#     json.dump(eval_results, f, indent=2, ensure_ascii=False)

# print("🎉 All Q&A pairs saved to test.json and eval.json")



















import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import torch
from tqdm import tqdm

# Path to the folder containing JSON files
input_folder = "/home/oem/Music/temp1/data/GPT_Greetings"  # Update this with your folder path
output_folder = "/home/oem/Music/temp1/data/GPT_Greetings/augmented"  # Output folder for results

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the model and tokenizer
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

# Iterate over all files in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".json"):  # Process only .json files
        print("################: this is the file name: ", filename)
        file_path = os.path.join(input_folder, filename)

        # Load data from each JSON file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Run QA generation for the current file
        results = []
        eval_results = []
        for i, item in tqdm(enumerate(data)):
            messages = [
                {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 5 new question that asks for the same information in a different way without changing the meaning. do not change the ansewr. only response in the following format ,\nnew-question:\nanswer: \nnew-question:\nanswer: \nnew-question:\nanswer:"},
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
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
            generated_text = outputs[0]["generated_text"][len(prompt):]
            print("##################", generated_text)

            qa_pairs = []
            qa_blocks = re.findall(
                r"new-question:\s*(.*?)\nanswer:\s*(.*?)(?=\nnew-question:|\Z)",
                generated_text.strip(), re.DOTALL
            )

            # Append original QA pair for evaluation
            qa_pairs.append({"question": item["question"], "answer": item["answer"]})

            # Take one QA pair for eval and rest for the test
            if qa_blocks:
                q_eval, a_eval = qa_blocks[0]
                eval_results.append({
                    "question": q_eval.strip(),
                    "answer": a_eval.strip()
                })
                for q, a in qa_blocks[1:]:
                    qa_pairs.append({
                        "question": q.strip(),
                        "answer": a.strip()
                    })

                results.extend(qa_pairs)

            print("Extracted QA blocks:", qa_blocks)
            print(f"✅ Processed paragraph {i+1} in {filename}")

        # Save the augmented data to a new file
        augmented_filename = os.path.join(output_folder, f"augmented_{filename}")
        with open(augmented_filename, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save evaluation data to a separate file
        eval_filename = os.path.join(output_folder, f"eval_{filename}")
        with open(eval_filename, "w") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        print(f"🎉 All Q&A pairs saved for {filename} to {augmented_filename} and {eval_filename}")

print("🎉 All JSON files processed successfully!")
