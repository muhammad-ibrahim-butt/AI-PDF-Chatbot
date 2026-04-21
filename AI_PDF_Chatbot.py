import streamlit as st
import os
import hashlib
import requests
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
CHAT_MODEL = "llama-3.1-8b-instant"

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

pdfs_directory = "pdfs/"
os.makedirs(pdfs_directory, exist_ok=True)


# ---- Local Embeddings (sentence-transformers) ----
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(texts):
    model = load_embed_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def get_embedding(text):
    return get_embeddings([text])[0]


# ---- Groq Chat (Llama 3.1 via OpenAI-compatible API) ----
def generate_answer(question, context):
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY. Set it in .env and restart the app."

    prompt = template.format(question=question, context=context)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    body = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=body, timeout=30)
        result = resp.json()
    except requests.RequestException as e:
        return f"Network error: {e}"

    err = result.get("error")
    if err:
        return f"API error: {err.get('message', str(err))}"

    choices = result.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return "Sorry, I couldn't generate a response."


# ---- Text Handling ----
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    return PDFPlumberLoader(file_path).load()


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


# ---- Indexing ----
def index_documents(chunks, progress_bar=None):
    texts = [c.page_content for c in chunks]
    embeddings = get_embeddings(texts)
    if progress_bar:
        progress_bar.progress(1.0)
    return [
        {"content": t, "embedding": e}
        for t, e in zip(texts, embeddings)
    ]


# ---- Retrieval ----
def retrieve_relevant_docs(store, query, top_k=3):
    if not store:
        return []

    query_emb = get_embedding(query)
    doc_embs = np.array([d["embedding"] for d in store])
    sims = cosine_similarity([query_emb], doc_embs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [store[i]["content"] for i in top_indices]


# ---- Streamlit App ----
if not GROQ_API_KEY:
    st.warning("Set the **GROQ_API_KEY** environment variable in `.env`, then restart Streamlit.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("indexed_hash") != file_hash:
        upload_pdf(uploaded_file)
        docs = load_pdf(pdfs_directory + uploaded_file.name)
        chunks = split_text(docs)

        st.info(f"Indexing {len(chunks)} chunks…")
        progress = st.progress(0.0)
        store = index_documents(chunks, progress_bar=progress)
        progress.empty()

        if store:
            st.session_state["documents_store"] = store
            st.session_state["indexed_hash"] = file_hash
            st.success(f"PDF processed and indexed ({len(store)} chunks).")
        else:
            st.error("Could not build embeddings.")
    else:
        st.success(
            f"PDF already indexed ({len(st.session_state.get('documents_store', []))} chunks). Ask a question below."
        )

    question = st.chat_input("Ask a question about the PDF")
    if question:
        st.chat_message("user").write(question)
        store = st.session_state.get("documents_store", [])
        with st.spinner("Thinking…"):
            relevant_contexts = retrieve_relevant_docs(store, question)
            context = "\n\n".join(relevant_contexts)
            answer = generate_answer(question, context)
        st.chat_message("assistant").write(answer)
