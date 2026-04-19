
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)


dic=  {
      "context": "What did these men and women see in him? He was not a preacher, nor a teacher. He was just a man who lived by his convictions and the tenets of his religion. What did he give them? I now think that it was his mere presence that calmed them and gave them hope. He said prayers for them, and many people would offer him bowls of water. He would dip his fingertips in them and say a prayer, after which the water would be taken away to be given to the sick. Later, many of these people would come to the house and thank him for having cured their near and dear ones.Why did he do this? And where did he get the peacefulness and generosity of heart to talk to people, comfort them and pray for them, in the midst of the busyness of his own life? He was a humble boat owner. Life was certainly not easy for him, what with finding the best ways to make ends meet in a tiny temple town cut off from the mainland. Yet, never once did I see my father turn away anyone who wanted to unburden himself by talking to him.",
      "question": "What was your father's occupation?",
      "answer":"My father was a humble boat owner."
  } 

print(dic["question"])
# paragraph= "Today I try to imagine him walking on those quiet roads, long before the day made its many demands on him. Ours was a large family and I am sure there were many pressures on him to see to our needs. But at that hour, I think of him listening intently to the sea, to the ever-present ravens and other birds that swooped and flew all around, woken up by the rising sun like him. Perhaps he said his prayers to himself as he walked, or thought of his family with a calm, uncluttered early morning mind. I never did ask him what went through his mind on this long daily walk—for when does a young boy really have the time to reflect in this way about his father? But I was always sure that the morning walk added something to his personality, an element of calm that was apparent even to strangers. My father was not a person with much formal education; neither did he ever acquire much wealth in his long lifetime. Yet, he was one of the wisest, truly generous men I have had the fortune of knowing. Our mosque was the focal point of the locality, and my father was the man everyone turned to in their hour of need. They believed that he was truly connected to God. I remember going to the mosque to say my prayers with him. He made sure we never missed any of our prayers and neither did it enter our minds to shirk this duty. After saying our namaz, when we would step out on to the road, groups of people would inevitably be there, waiting to talk to him and share their worries with him."

# messages = [
#     {"role": "system", "content": "You are Dr. A.P.J. Abdul Kalam, former President of India and a humble, wise teacher. Based only on the paragraph below, you are given a question-answer pair that a curious student has asked you. Your task is to rephrase the question and answer in a natural and thoughtful way, while keeping the original meaning intact. Ensure the rephrased question still clearly refers to the context, and the answer reflects your calm, inspiring, and encouraging tone. Keep both concise and to the point—like in real conversations or interviews. Reflect Dr. Kalam’s values of simplicity, positivity, and clarity."},
#     {"role": "user", "content": f'context:{dic["context"]}\n question:{dic["question"]}\n answer:{dic["answer"]}'},
# ]

# for augmentaion
messages = [
    {"role": "system", "content": "You are an AI trained to help improve question-answering datasets. Given a context, a question, and its answer, generate 3 new question that asks for the same information in a different way without changing the meaning. Also, rephrase the answer appropriately."},
    {"role": "user", "content": f'question:{dic["question"]}\n answer:{dic["answer"]}'},
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














# import re

# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import pipeline
# import torch
# from tqdm import tqdm

# # Load data
# with open("qa_context_data.json", "r") as f:
#     data = json.load(f)

# # Extract full context text
# full_context = "\n".join([entry["context"] for entry in data if "context" in entry])

# # Function to extract unique paragraphs using TF-IDF similarity
# def extract_unique_paragraphs(text, threshold=0.85):
#     paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
#     unique = []
#     vectorizer = TfidfVectorizer().fit(paragraphs)
#     tfidf_matrix = vectorizer.transform(paragraphs)

#     for i, para in enumerate(paragraphs):
#         sim = (tfidf_matrix[i] @ tfidf_matrix[:i].T).toarray()
#         if sim.shape[1] == 0 or sim.max() < threshold:
#             unique.append(para)
#     return unique

# # Extract unique paragraphs
# unique_paragraphs = extract_unique_paragraphs(full_context)

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

# def no_of_question(char_count):
#     if char_count< 500:
#     	no_of_ques = 4
#     	return no_of_ques

#     if char_count >= 500 and char_count< 600:
#     	no_of_ques = 5
#     	return no_of_ques

#     if char_count >= 600 and char_count< 700:
#     	no_of_ques = 6
#     	return no_of_ques

#     if char_count >= 700 and char_count< 800:
#     	no_of_ques = 7
#     	return no_of_ques

#     if char_count >= 800 and char_count< 900:
#     	no_of_ques = 8
#     	return no_of_ques

#     if char_count >= 900 and char_count< 1000:
#     	no_of_ques = 9
#     	return no_of_ques

#     if char_count >= 1000 and char_count< 1100:
#     	no_of_ques = 10
#     	return no_of_ques

#     if char_count >= 1100 and char_count< 1200:
#     	no_of_ques = 11
#     	return no_of_ques
#     if char_count >= 1200:
#     	no_of_ques = 12
#     	return no_of_ques


# # Step 1: Load paragraphs from txt file
# with open("my_journey_v2.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# # Step 2: Remove page markers and split into paragraphs
# import re
# clean_text = re.sub(r"--- Page \d+ ---", "", raw_text)
# paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]




# # Run QA generation
# results = []

# for i, paragraph in tqdm(enumerate(paragraphs)):
#     char_count = len(paragraph.replace(" ", ""))
#     print(f"Paragraph {i+1} length (no spaces): {char_count} characters")

#     no_of_ques = no_of_question(char_count)
#     print(paragraph)
#     print(no_of_ques)

#     messages = [
#         {
#             "role": "system",
#             "content":f"You are Dr. A.P.J. Abdul Kalam, former President of India and a humble, wise teacher. Based only on the paragraph below, generate {no_of_ques} natural and thoughtful question-answer pairs, as if a curious student is asking you questions and you are replying in your calm, inspiring, and encouraging tone. Keep the answers concise and to the point—just like in real conversations or interviews. Avoid long explanations. Reflect Dr. Kalam's values of simplicity, positivity, and clarity in both questions and answers."
#         },
#         {
#             "role": "user",
#             "content": f"paragraph: {paragraph}"
#         }
#     ]

#     prompt = pipeline.tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         prompt,
#         max_new_tokens=1024,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     print(outputs[0]["generated_text"][len(prompt):])
#     generated_text = outputs[0]["generated_text"][len(prompt):]
#     qa_pairs = []
#     qa_blocks = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", generated_text.strip(), re.DOTALL)
#     for q, a in qa_blocks:
#     	qa_pairs.append({
#     		"context": paragraph,
#     		"question": q.strip(),
#     		"answer": a.strip()
#     		})

#     results.extend(qa_pairs)

#     # results.append({
#     #     "paragraph": paragraph,
#     #     "qa_pairs": generated_text.strip()
#     # })


#     print(f"✅ Processed paragraph {i+1}/{len(paragraphs)}")

# # Save to JSON
# # with open("qa_output_llama3_updated_prompt.json", "w") as f:
# #     json.dump(results, f, indent=2)

# with open("qa_output_llama3_updated_prompt_no_of_ques_txt_para.json", "w") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)


# print("🎉 All Q&A pairs saved to qa_output.json")





 # "Consider you are Dr. A.P.J. Abdul Kalam, former President of India. Only based on the paragraph below, generate 10 thoughtful question-answer pairs by considering someone asking questions to you and you are answering as Dr. Kalam. Always answer in a calm, wise, and encouraging tone."



# "You are Dr. A.P.J. Abdul Kalam, former President of India and a humble, wise teacher. Based only on the paragraph below, generate 10 natural and thoughtful question-answer pairs, as if a curious student is asking you questions and you are replying in your calm, inspiring, and encouraging tone. Keep the answers concise and to the point—just like in real conversations or interviews. Avoid long explanations. Reflect Dr. Kalam's values of simplicity, positivity, and clarity in both questions and answers."
