import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline as hf_pipeline

st.set_page_config(page_title="RAG Demo", page_icon="🔍", layout="wide")

st.title("🔍 RAG Demo")
st.caption("Upload a document and ask questions — runs fully on-device, no API key needed.")


# ── Model loading (cached across sessions) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    pipe = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256,
        do_sample=False,
        device="cpu",
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return embeddings, llm


embeddings, llm = load_models()


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


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

# Re-index only when file or settings change
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

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = qa_chain.invoke({"query": query})

        answer = result["answer"]
        sources = result["source_documents"]

        st.markdown(answer)

        with st.expander(f"📄 {len(sources)} retrieved chunk(s)"):
            for i, doc in enumerate(sources, 1):
                label = f"Chunk {i}"
                if "page" in doc.metadata:
                    label += f" — page {doc.metadata['page'] + 1}"
                st.markdown(f"**{label}**")
                st.markdown(f"> {doc.page_content.strip()}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
