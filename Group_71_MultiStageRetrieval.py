import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
# python -m pip install faiss-cpu rank_bm25 sentence-transformers transformers beautifulsoup4
install("faiss-cpu")
install("rank_bm25")
install("sentence-transformers")
install("transformers")
# install("beautifulsoup4")

# python -m pip install transformers[sentencepiece]  # Ensure sentencepiece is installed for tokenization
# install("transformers[sentencepiece]")

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

# python -m pip install pdfplumber
install("pdfplumber")

# from google.colab import files
import pdfplumber

def preprocess_pdfs(pdf_paths):
    text_chunks = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                # Split into chunks
                for i in range(0, len(text), 500):
                    chunk = text[i:i+500]
                    text_chunks.append(chunk)
    return text_chunks

# Upload and Load Financial Documents from HTML
# print("Please upload financial statements in PDF format")
# pdf_paths = files.upload()
pdf_paths = ['msft-20230630_10k_2023.pdf','msft-20240630_10k_2024.pdf']
chunks = preprocess_pdfs(pdf_paths)

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, convert_to_tensor=False)

# Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Load SLM
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

def basic_rag(query):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = " ".join(retrieved_chunks)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def advanced_rag(query):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=10)  # Retrieve top 10
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Re-rank with cross-encoder
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_chunk = retrieved_chunks[ranked_indices[0]]  # Take top-ranked chunk

    # Generate response
    input_text = f"Question: {query}\nContext: {top_chunk}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence = scores[ranked_indices[0]]  # Use cross-encoder score as confidence
    return answer, confidence

from rank_bm25 import BM25Okapi

# Tokenize chunks for BM25
tokenized_chunks = [chunk.split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def advanced_rag(query):
    # Stage 1: Coarse retrieval with BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    coarse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
    coarse_chunks = [chunks[i] for i in coarse_indices]

    # Stage 2: Fine retrieval with embeddings
    coarse_embeddings = embedder.encode(coarse_chunks)
    coarse_index = faiss.IndexFlatL2(dimension)
    coarse_index.add(coarse_embeddings)
    query_embedding = embedder.encode([query])
    D, I = coarse_index.search(query_embedding, k=5)  # Top 5 from coarse set
    final_chunks = [coarse_chunks[i] for i in I[0]]

    # Generate response
    # Shorten the context to prevent exceeding max_length
    context = " ".join(final_chunks)[:200] # Limit context to 200 characters
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Confidence score (normalized BM25 score + embedding distance)
    confidence = (bm25_scores[coarse_indices[0]] / max(bm25_scores)) * 0.5 + (1 - D[0][0]) * 0.5
    return answer, confidence

# python -m pip install streamlit
install("streamlit")


import streamlit as st

st.title("Financial RAG Chatbot - Group 71")
query = st.text_input("Ask a financial question:")
if query:
    answer, confidence = advanced_rag(query)
    st.write(f"Answer: {answer}")
    st.write(f"Confidence Score: {confidence:.2f}")


def guardrail_filter(answer):
    financial_keywords = ["revenue", "profit", "loss", "income", "expense", "balance"]
    if not any(keyword in answer.lower() for keyword in financial_keywords):
        return "Sorry, I couldn’t provide a relevant financial answer."
    return answer

def advanced_rag_with_guardrail(query):
    answer, confidence = advanced_rag(query)
    filtered_answer = guardrail_filter(answer)
    return filtered_answer, confidence

test_questions = [
    "What was Microsoft’s revenue in 2023?",
    "How did Microsoft perform financially compared to Apple?",
    "What is the capital of France?"
]

for q in test_questions:
    answer, confidence = advanced_rag_with_guardrail(q)
    print(f"Q: {q}\nA: {answer}\nConfidence: {confidence:.2f}\n")
