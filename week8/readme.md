# 💬 Loan Rule RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot built using your own CSV rules (loan policies) and a fast open-source language model (`flan-t5-base`). Ask natural-language questions and receive intelligent answers based on your custom rule base.

---

## 🚀 Features

- 💾 Use your own CSV with loan rules (in plain English)
- 🔍 FAISS-based vector search for fast semantic retrieval
- 🤖 Generative answers using Hugging Face’s `flan-t5-base`
- 🧠 Supports CPU (no GPU required)
- 🖥️ Clean Streamlit UI for real-time interaction
- 🛠️ Fully offline/local setup supported

---

## 📁 Folder Structure

rag_chatbot/
├── app.py # Streamlit UI
├── rag_chatbot.py # Build & query RAG model
├── loan_rules.csv # Your rules (1 column: 'text')
├── rules_index.faiss # FAISS index (generated)
├── rules.pkl # Chunks cache (generated)
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 📝 Sample `loan_rules.csv`

Your CSV must have a column named `text` like this:

```csv
text
Applicants with a credit history of 1 are usually approved.
Loan applications with an income below 2000 are often rejected.
Self-employed applicants must show consistent income to qualify.
Urban area applicants are slightly more likely to be approved than rural.
🧪 Getting Started
1. Clone the project
bash
Copy
Edit
git clone <your-repo-url>
cd rag_chatbot
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, run:

bash
Copy
Edit
pip install pandas numpy faiss-cpu sentence-transformers transformers sentencepiece streamlit
3. Build RAG index
bash
Copy
Edit
python rag_chatbot.py --mode build --csv loan_rules.csv
4. Run the Streamlit UI
bash
Copy
Edit
streamlit run app.py
❓Example Questions to Try
What happens if someone is self-employed?

Does credit history affect approval?

Is urban property better than rural?

What if the applicant has no dependents?

⚡ Performance Notes
Embedding model: all-MiniLM-L6-v2 (small, fast)

Generator model: google/flan-t5-base (lightweight, CPU-friendly)

All models are downloaded on first run and cached.

📌 Credits
Hugging Face Transformers

Sentence Transformers

FAISS

Streamlit

📬 Contact
Created with ❤️ for fast, rule-based document Q&A.
Feel free to fork and adapt this for your own policy, FAQ, or HR rule chatbots!

yaml
Copy
Edit

---

Let me know if you want me to:
- Auto-generate a `requirements.txt`
- Zip the project folder
- Help you deploy it to Streamlit Cloud or Hugging Face Spaces