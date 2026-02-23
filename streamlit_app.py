import os
import tempfile

import streamlit as st
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="RAG Demo", page_icon="🔍", layout="wide")

st.title("🔍 RAG Demo")
st.caption("Upload a document and ask questions — runs fully on-device, no API key needed.")

MODEL_NAME = "google/flan-t5-small"


# ── Model loading (cached across sessions) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return embeddings, tokenizer, model


embeddings, tokenizer, gen_model = load_models()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size", 128, 512, 256, step=64)
    chunk_overlap = st.slider("Chunk overlap", 0, 128, 32, step=16)
    top_k = st.slider("Chunks to retrieve", 1, 4, 2)
    st.divider()
    st.caption("LLM: `flan-t5-small`  \nEmbeddings: `all-MiniLM-L6-v2`")
    st.caption("Built with [LangChain](https://python.langchain.com) + [FAISS](https://faiss.ai)")


# ── Document indexing ─────────────────────────────────────────────────────────
def build_index(file_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int):
    suffix = ".pdf" if filename.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


# ── Answer generation ─────────────────────────────────────────────────────────
def generate_answer(query: str, vectorstore, top_k: int):
    docs = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True), docs


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

file_key = (uploaded.name, uploaded.size, chunk_size, chunk_overlap)
if st.session_state.get("file_key") != file_key:
    with st.spinner("Indexing document…"):
        vectorstore, n_chunks = build_index(
            uploaded.read(), uploaded.name, chunk_size, chunk_overlap
        )
    st.session_state.file_key = file_key
    st.session_state.vectorstore = vectorstore
    st.session_state.n_chunks = n_chunks
    st.session_state.messages = []

st.success(f"Document indexed — {st.session_state.n_chunks} chunks.")


# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, sources = generate_answer(query, st.session_state.vectorstore, top_k)

        st.markdown(answer)

        with st.expander(f"📄 {len(sources)} retrieved chunk(s)"):
            for i, doc in enumerate(sources, 1):
                label = f"Chunk {i}"
                if "page" in doc.metadata:
                    label += f" — page {doc.metadata['page'] + 1}"
                st.markdown(f"**{label}**")
                st.markdown(f"> {doc.page_content.strip()}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
