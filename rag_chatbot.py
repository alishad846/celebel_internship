import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

### 1. Load CSV with rule chunks
def load_chunks(csv_path):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column")
    return df['text'].dropna().astype(str).tolist()

### 2. Build FAISS index
def build_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    faiss.write_index(index, "rules_index.faiss")
    with open("rules.pkl", "wb") as f:
        pickle.dump(chunks, f)

### 3. Ask a question
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def format_prompt(context, question):
    return f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"

def ask_question(question, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("rules_index.faiss")
    with open("rules.pkl", "rb") as f:
        chunks = pickle.load(f)

    q_vec = model.encode([question])
    _, I = index.search(np.array(q_vec), top_k)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = format_prompt(context, question)
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return result

### 4. CLI Interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "ask"], required=True)
    parser.add_argument("--csv", help="Path to your CSV file")
    parser.add_argument("--question", help="Ask a question")
    args = parser.parse_args()

    if args.mode == "build":
        if not args.csv: raise Exception("CSV path is required.")
        chunks = load_chunks(args.csv)
        build_index(chunks)
        print(f"✅ RAG index built from {args.csv} with {len(chunks)} rules.")
    elif args.mode == "ask":
        if not args.question: raise Exception("You must enter a question.")
        answer = ask_question(args.question)
        print("\n📌 Answer:\n", answer)
