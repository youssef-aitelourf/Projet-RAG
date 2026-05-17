from sentence_transformers import CrossEncoder
from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerRAG(BaseRAG):
    """
    Reranker RAG (two-stage retrieval):
    1. Bi-encoder fast retrieval: fetch top_k_fetch candidates (large).
    2. Cross-encoder reranking: score every (query, doc) pair precisely.
    3. Keep only top_k after reranking.
    Higher precision at the cost of extra local inference per query.
    """
    name = "reranker_rag"

    def __init__(self, cfg: Config, collection: str = "reranker"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)
        self._reranker = CrossEncoder(_RERANKER_MODEL)

    def index(self, chunks: list[Chunk]) -> None:
        self._store.reset()
        self._store.add(chunks)

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        candidates = self._store.search(q_emb, self._cfg.top_k_fetch)
        pairs = [(question, d["text"]) for d in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in ranked[: self._cfg.top_k]]
        context = self._format_context(top_docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(answer=answer, retrieved_docs=top_docs, queries_used=[question])
