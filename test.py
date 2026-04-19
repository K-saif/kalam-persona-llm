# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load your fine-tuned model and tokenizer
# model_dir = "/home/oem/Music/temp1/outputs/checkpoint-650"  # adjust if needed
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForCausalLM.from_pretrained(model_dir)
# model.eval()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Inference loop
# while True:
#     print("\n--- New Input ---")
#     prompt = "consider you are APJ abdul kalam response t"
#     if prompt.lower() in ['exit', 'quit']:
#         break
#     context = input("Enter context: ")
#     question = input("Enter question: ")

#     # Format input like the training format
#     input_text = f"prompt: {prompt} context: {context} question: {question}"

#     # Tokenize and move to device
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)

#     # Generate answer
#     output = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     # Decode and print only the generated answer
#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
#     generated_answer = decoded.replace(input_text, "").strip()
#     print("\n🧠 Model's Answer:\n", generated_answer)





# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import pickle

# # Load RAG context
# with open("context_paragraphs.pkl", "rb") as f:
#     context_paragraphs = pickle.load(f)
# context_embeddings = np.load("context_embeddings.npy")

# # Load embedding model (same one used earlier)
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Build NearestNeighbors index
# nn = NearestNeighbors(n_neighbors=1, metric='cosine')
# nn.fit(context_embeddings)

# # Load fine-tuned Gemma model
# model_name = "/home/oem/Music/temp1/outputs/checkpoint-650"  # Replace this
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# # Function to retrieve relevant context
# def retrieve_context(question):
#     q_emb = embed_model.encode([question])
#     dist, idx = nn.kneighbors(q_emb)
#     return context_paragraphs[idx[0][0]]

# def generate_answer(question):
#     context = retrieve_context(question)
#     # context = "Today I try to imagine him walking on those quiet roads, long before the day made its many demands on him. Ours was a large family and I am sure there were many pressures on him to see to our needs. But at that hour, I think of him listening intently to the sea, to the ever-present ravens and other birds that swooped and flew all around, woken up by the rising sun like him. Perhaps he said his prayers to himself as he walked, or thought of his family with a calm, uncluttered early morning mind. I never did ask him what went through his mind on this long daily walk—for when does a young boy really have the time to reflect in this way about his father? But I was always sure that the morning walk added something to his personality, an element of calm that was apparent even to strangers. My father was not a person with much formal education; neither did he ever acquire much wealth in his long lifetime. Yet, he was one of the wisest, truly generous men I have had the fortune of knowing. Our mosque was the focal point of the locality, and my father was the man everyone turned to in their hour of need. They believed that he was truly connected to God. I remember going to the mosque to say my prayers with him. He made sure we never missed any of our prayers and neither did it enter our minds to shirk this duty. After saying our namaz, when we would step out on to the road, groups of people would inevitably be there, waiting to talk to him and share their worries with him."
#     prompt = f"""<start_of_turn>user Below is a question based on a passage. Answer it based on the context provided.\n\nContext: {context}\n\nQuestion: {question} <end_of_turn>\n<start_of_turn>model"""
    
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     output = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")  # Stop generation at <end_of_turn>
#     )
#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     # Optional: Strip off the prompt from the output if it leaks
#     return decoded.replace(prompt, "").strip()


# while True:
#     # 🔍 Try it out
#     question = input("question: ")
#     response = generate_answer(question)
#     print("🧠 Response:\n", response)









import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

# -----------------------------
# Load Precomputed Paragraphs and Embeddings
# -----------------------------
with open("RAG/qa_merged_v1.pkl", "rb") as f:
    context_paragraphs = pickle.load(f)

context_embeddings = np.load("RAG/qa_merged_v1.npy")

# -----------------------------
# Load Embedding Model
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Build Nearest Neighbors Index
# -----------------------------
nn = NearestNeighbors(n_neighbors=5, metric='cosine')  # top-5 for flexibility
nn.fit(context_embeddings)

# -----------------------------
# Load Fine-tuned Gemma Model
# -----------------------------
model_path = "/home/oem/Music/temp1/output/llama3_qa_10000/checkpoint-10000"  # Your fine-tuned path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# -----------------------------
# Retrieve Top Matching Contexts
# -----------------------------
# def retrieve_context(question, top_k=3, threshold=0.4):
#     q_emb = embed_model.encode([question])
#     distances, indices = nn.kneighbors(q_emb, n_neighbors=top_k)

#     relevant_contexts = []
#     for i, dist in zip(indices[0], distances[0]):
#         if dist < threshold:
#             relevant_contexts.append(context_paragraphs[i])

#     if not relevant_contexts:
#         relevant_contexts.append(context_paragraphs[indices[0][0]])

#     return "\n\n".join(relevant_contexts)

def retrieve_context(question, top_k=3):
    q_emb = embed_model.encode([question])
    distances, indices = nn.kneighbors(q_emb, n_neighbors=top_k)
    
    # 🧠 Just take top_k paragraphs directly — no threshold filtering
    return "\n\n".join([context_paragraphs[i] for i in indices[0]])



# -----------------------------
# Generate Answer
# -----------------------------
# def generate_answer(question):
#     context = retrieve_context(question)

#     prompt = f"""<start_of_turn>user Below is a question based on a passage. Answer it based on the context provided.\n\nContext: {context}\n\nQuestion: {question} <end_of_turn>\n<start_of_turn>model"""

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
#     )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)

#     return decoded.replace(prompt, "").strip()


def generate_answer(question):
    # context = retrieve_context(question)
    
    # prompt = f"""<start_of_turn>user You are Dr. A.P.J. Abdul Kalam, a wise and humble teacher. Based on the passage below, answer the student's question in a calm, thoughtful, and inspiring tone. Keep the response concise and encouraging.\n\nContext: {context}\n\nQuestion: {question} <end_of_turn>\n<start_of_turn>model"""

    # prompt = f"""<start_of_turn>user You are Dr. A.P.J. Abdul Kalam, a wise and humble teacher. Answer the student's question in a calm, thoughtful, and inspiring tone. Keep the response concise and encouraging.\n\nQuestion: {question} <end_of_turn>\n<start_of_turn>model"""

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are Dr. A.P.J. Abdul Kalam, a wise and humble teacher. Based on the passage below, answer the student's question in a calm, thoughtful, and inspiring tone. Keep the response concise and encouraging.<|eot_id|><|start_header_id|>user<|end_header_id|> Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|> Assistant <|eot_id|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,  # ← disable randomness
        num_beams=1,      # ← simple greedy decoding
        eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded.replace(prompt, "").strip()


# -----------------------------
# Interactive Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        question = input("\n❓ Question: ")
        response = generate_answer(question)
        print("\n🧠 Answer:\n", response)

