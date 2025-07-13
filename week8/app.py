import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load index and chunks
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("rules_index.faiss")
    with open("rules.pkl", "rb") as f:
        chunks = pickle.load(f)
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return model, index, chunks, generator

def ask_question(query, model, index, chunks, generator, top_k=3):
    q_vec = model.encode([query])
    _, I = index.search(np.array(q_vec), top_k)
    context = "\n".join([chunks[i] for i in I[0]])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return result

# UI
st.set_page_config(page_title="Loan Policy RAG Chatbot", layout="centered")
st.title("💬 Loan Rule Q&A Chatbot")
st.markdown("Ask anything based on your custom loan policy rules.")

query = st.text_input("🔍 Enter your question:")
if query:
    with st.spinner("Generating answer..."):
        model, index, chunks, generator = load_model_and_data()
        answer = ask_question(query, model, index, chunks, generator)
        st.success("✅ Answer:")
        st.write(answer)
