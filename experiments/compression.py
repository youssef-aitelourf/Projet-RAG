import re

import numpy as np

from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult

_COMPRESS_PROMPT = """\
Extract ONLY the sentences from the passage that are directly relevant to answering the question.
Preserve the original wording exactly. If no sentences are relevant, output "IRRELEVANT".

Question: {question}
Passage: {passage}

Relevant sentences:"""

_EXTRACTIVE_MIN_SCORE = 0.25


class CompressionRAG(BaseRAG):
    """
    Contextual Compression RAG:
    - extractive mode (default): sentence similarity via embeddings, no extra LLM calls
    - llm mode: LLM extracts relevant sentences per candidate chunk
  """
    name = "compression_rag"

    def __init__(self, cfg: Config, collection: str = "compression"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)
        self._mode = cfg.compression_mode

    def index(self, chunks: list[Chunk], fresh: bool = False) -> None:
        self._prepare_vector_index(self._store, chunks, fresh)

    def _compress_llm(self, question: str, doc: dict) -> dict | None:
        compressed = self._llm.ask(
            _COMPRESS_PROMPT.format(question=question, passage=doc["text"]),
            temperature=0.0,
        )
        if compressed.strip().upper().startswith("IRRELEVANT"):
            return None
        return {**doc, "text": compressed.strip()}

    def _compress_extractive(self, question: str, doc: dict) -> dict | None:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"].strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return None
        q_emb = self._embedder.embed_one(question)
        s_embs = self._embedder.embed(sentences)
        scores = s_embs @ q_emb
        selected = [
            sent for sent, score in zip(sentences, scores)
            if score >= _EXTRACTIVE_MIN_SCORE
        ]
        if not selected:
            best_idx = int(np.argmax(scores))
            selected = [sentences[best_idx]]
        return {**doc, "text": " ".join(selected)}

    def _compress(self, question: str, doc: dict) -> dict | None:
        if self._mode == "llm":
            return self._compress_llm(question, doc)
        return self._compress_extractive(question, doc)

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        fetch_k = self._cfg.top_k if self._mode == "llm" else self._cfg.top_k * 2
        candidates = self._store.search(q_emb, fetch_k)

        compressed_docs: list[dict] = []
        for doc in candidates:
            result = self._compress(question, doc)
            if result:
                compressed_docs.append(result)
            if len(compressed_docs) >= self._cfg.top_k:
                break

        final_docs = compressed_docs if compressed_docs else candidates[: self._cfg.top_k]
        context = self._format_context(final_docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(
            answer=answer,
            retrieved_docs=final_docs,
            queries_used=[question],
        )
