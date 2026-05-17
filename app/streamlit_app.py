"""Streamlit demo UI for Projet-RAG experiments."""

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.loader import load_file
from experiments.naive import NaiveRAG
from experiments.hyde import HyDERAG
from experiments.fusion import RAGFusion
from experiments.reranker import RerankerRAG
from experiments.hybrid import HybridRAG
from experiments.parent_child import ParentChildRAG
from experiments.compression import CompressionRAG

EXPERIMENTS = {
    "naive": NaiveRAG,
    "hyde": HyDERAG,
    "fusion": RAGFusion,
    "reranker": RerankerRAG,
    "hybrid": HybridRAG,
    "parent_child": ParentChildRAG,
    "compression": CompressionRAG,
}


@st.cache_resource
def get_experiment(name: str):
    cfg = Config()
    return EXPERIMENTS[name](cfg, collection=f"demo_{name}")


st.set_page_config(page_title="Projet-RAG", page_icon="🔍", layout="wide")
st.title("Projet-RAG — Experiment Playground")

with st.sidebar:
    st.header("Settings")
    exp_name = st.selectbox("RAG method", list(EXPERIMENTS.keys()), index=0)
    strategy = st.selectbox("Chunking", ["recursive", "fixed", "sentence"], index=0)
    fresh_index = st.checkbox("Fresh index (reset)", value=True)

    uploaded = st.file_uploader("Upload document (.txt, .md)", type=["txt", "md"])
    use_corpus = st.checkbox("Use default corpus (data/samples/)", value=True)

    if st.button("Index documents", type="primary"):
        exp = get_experiment(exp_name)
        chunks = []
        if use_corpus:
            from src.loader import load_dir
            chunks.extend(load_dir(ROOT / "data" / "samples"))
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            chunks.extend(load_file(tmp_path, strategy=strategy))
        if not chunks:
            st.error("No documents to index.")
        else:
            with st.spinner(f"Indexing {len(chunks)} chunks…"):
                exp.index(chunks, fresh=fresh_index)
            st.success(f"Indexed {len(chunks)} chunks into demo_{exp_name}")

question = st.text_input("Your question", placeholder="How does HyDE work?")
if st.button("Ask", type="primary") and question.strip():
    exp = get_experiment(exp_name)
    with st.spinner("Running RAG pipeline…"):
        result = exp.query(question.strip())

    st.subheader("Answer")
    st.write(result.answer)

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{result.latency:.2f}s")
    col2.metric("Docs retrieved", len(result.retrieved_docs))
    col3.metric("Queries used", len(result.queries_used))

    st.subheader("Retrieved passages")
    for i, doc in enumerate(result.retrieved_docs, 1):
        score = doc.get("score", 0)
        source = doc.get("source", "?")
        with st.expander(f"[{i}] {source} — score {score:.3f}"):
            st.write(doc["text"])

    with st.expander("Queries used"):
        for q in result.queries_used:
            st.code(q)
