import json

# Load your existing QA data
with open("RAG/qa_merged_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract unique contexts using a set for deduplication
unique_contexts = list({entry["context"].strip() for entry in data if "context" in entry})

print(f"Total unique contexts found: {len(unique_contexts)}")


import pickle

# Save unique contexts
with open("RAG/qa_merged_v1.pkl", "wb") as f:
    pickle.dump(unique_contexts, f)



from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all context paragraphs
context_embeddings = embed_model.encode(unique_contexts, batch_size=32, show_progress_bar=True)

# Save embeddings
np.save("RAG/qa_merged_v1.npy", context_embeddings)











Context: Teachers and mentors come at various stages into our lives. As a child, I looked up to my parents and my teachers. Then my dear friend and brother- in-law, Ahmed Jalalluddin, guided me in the crucial years when I turned from a child into a man. And as my career was beginning, I was immensely lucky to come in the orbit of a man such as Dr Vikram Sarabhai.

A few remarkable people have appeared at critical times in my life and proceeded to mould or reorient my ways of thinking; sometimes they have even changed the course of my life. To these mentors I am always grateful and remember them more and more each day. Now, if I could have all the time in the world, I know what I would do; I would spend time in remembering these people who shaped my life. They are like the sun that warms the face and the winds that embrace. One such person in my life was Ahmed Jalalluddin.

In many ways my real education began after I left Rameswaram for high school at Ramanathapuram. As I have written earlier, it was the first time I stepped out of the protective embrace of Rameswaram, my mother and everything else that was familiar. I was very much a shy small-town boy then, afraid to speak out much. It was at Schwartz High School that I had my first brush with the wonders of science, and had it explained to me in a manner that set my mind alight. At that school there was a teacher called Reverend Iyadurai Solomon. He struck up a relationship of great openness and trust with me. In him, I found the guide that I needed to show me the path forward.
