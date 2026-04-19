

import re

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import torch
from tqdm import tqdm

# Load data
with open("data/qa_merged_v1.json", "r") as f:
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
#       no_of_ques = 4
#       return no_of_ques

#     if char_count >= 500 and char_count< 600:
#       no_of_ques = 5
#       return no_of_ques









# Run QA generation
results = []
eval_results = []
for i, item in tqdm(enumerate(data)):
    # char_count = len(paragraph.replace(" ", ""))
    # no_of_ques = no_of_question(char_count)

    # for augmentaion
    messages = [
        #for without context
        # {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 3 new answer that response for the same information in a different way without changing the meaning. write some answer in one word if it is applicable. only response in the following format ,\nnew-question:\nnew-answer:"},
        
        {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a question and its answer, generate 3 new question that asks for the same information in a different way without changing the meaning. Also, rephrase the answer appropriately. only response in the following format ,\nnew-question:\nnew-answer:"},
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
        top_p=0.9)
    generated_text = outputs[0]["generated_text"][len(prompt):]
    print(generated_text)
    
    qa_blocks = re.findall(
        r"new-question:\s*(.*?)\nnew-answer:\s*(.*?)(?=\nnew-question:|\Z)",
        generated_text.strip(), re.DOTALL
    )

    # Take one QA pair for eval and rest for test
    if qa_blocks:
        q_eval, a_eval = qa_blocks[0]
        eval_results.append({
            "question": q_eval.strip(),
            "answer": a_eval.strip()
        })
        for q, a in qa_blocks[1:]:
            results.append({
                "question": q.strip(),
                "answer": a.strip()
            })

    print("Extracted QA blocks:", qa_blocks)
    print(f"✅ Processed paragraph {i+1}")

# Save test.json
with open("data/augmentation_eval_qa_merged_v1.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Save eval.json
with open("data/augmentation_eval_qa_merged_v1_eval.json", "w") as f:
    json.dump(eval_results, f, indent=2, ensure_ascii=False)

print("🎉 All Q&A pairs saved to test.json and eval.json")
