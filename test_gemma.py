import torch
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from huggingface_hub import login
login(token="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", add_to_git_credential=False)


# -----------------------------
# Step 1: Load saved context and embeddings
# -----------------------------
with open("context_paragraphs.pkl", "rb") as f:
    context_paragraphs = pickle.load(f)

context_embeddings = np.load("context_embeddings.npy")

# -----------------------------
# Step 2: Set up retriever (Nearest Neighbors)
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
nn = NearestNeighbors(n_neighbors=3, metric="cosine")
nn.fit(context_embeddings)

# -----------------------------
# Step 3: Load the official Gemma 2B IT model with 4-bit quantization
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

# -----------------------------
# Step 4: Retrieve top-3 most relevant paragraphs for the input question
# -----------------------------
def retrieve_context(question, k=3):
    q_embedding = embed_model.encode([question])
    distances, indices = nn.kneighbors(q_embedding, n_neighbors=k)
    top_paragraphs = [context_paragraphs[i] for i in indices[0]]
    return "\n\n".join(top_paragraphs)

# -----------------------------
# Step 5: Format prompt and generate answer
# -----------------------------
# def generate_answer(question):
#     # context = retrieve_context(question)
#     context = "Today I try to imagine him walking on those quiet roads, long before the day made its many demands on him. Ours was a large family and I am sure there were many pressures on him to see to our needs. But at that hour, I think of him listening intently to the sea, to the ever-present ravens and other birds that swooped and flew all around, woken up by the rising sun like him. Perhaps he said his prayers to himself as he walked, or thought of his family with a calm, uncluttered early morning mind. I never did ask him what went through his mind on this long daily walk—for when does a young boy really have the time to reflect in this way about his father? But I was always sure that the morning walk added something to his personality, an element of calm that was apparent even to strangers. My father was not a person with much formal education; neither did he ever acquire much wealth in his long lifetime. Yet, he was one of the wisest, truly generous men I have had the fortune of knowing. Our mosque was the focal point of the locality, and my father was the man everyone turned to in their hour of need. They believed that he was truly connected to God. I remember going to the mosque to say my prayers with him. He made sure we never missed any of our prayers and neither did it enter our minds to shirk this duty. After saying our namaz, when we would step out on to the road, groups of people would inevitably be there, waiting to talk to him and share their worries with him."
#     prompt = f"""<start_of_turn>below is the paragraph and make question answer pairs based on the paragraph in such  .\n\nparagraph: {context}<end_of_turn>\n<start_of_turn>model"""

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     output = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
#     )

#     decoded = tokenizer.decode(output[0], skip_special_tokens=True)
#     return decoded.replace(prompt, "").strip()

# # -----------------------------
# # Step 6: Run Inference
# # -----------------------------
# if __name__ == "__main__":
#     while True:
#         question = input("❓ Question: ")
#         if question.lower() in ['exit', 'quit']:
#             break
#         answer = generate_answer(question)
#         print("🧠 Answer:\n", answer)




def generate_answer(paragraph):
    prompt = f"""
<start_of_turn>user
You are Dr. A.P.J. Abdul Kalam, former President of India and a visionary teacher. Based on the paragraph below, generate 2 thoughtful question-answer pairs. Your answers should reflect your values: humility, scientific spirit, faith, simplicity, and the ability to inspire youth. Always answer in a calm, wise, and encouraging tone — just as Dr. Kalam would.

Paragraph:
\"\"\"{paragraph}\"\"\"


<end_of_turn>
<start_of_turn>model"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()



if __name__ == "__main__":
    paragraph = """Today I try to imagine him walking on those quiet roads, long before the day made its many demands on him..."""
    while True:
        cmd = input("▶️ Generate Q&A? (y/n): ")
        if cmd.lower() in ["n", "exit", "quit"]:
            break
        result = generate_answer(paragraph)
        print("📚 Generated QA Pairs:\n", result)
