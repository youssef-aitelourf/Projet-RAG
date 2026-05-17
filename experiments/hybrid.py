from collections import defaultdict

from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.bm25_store import BM25Store
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult


class HybridRAG(BaseRAG):
    """
    Hybrid Search RAG (dense + sparse):
    1. Dense retrieval via cosine similarity on embeddings.
    2. Sparse retrieval via BM25 keyword matching.
    3. Merge both result lists with Reciprocal Rank Fusion.
    Fixes the semantic gap for exact-term queries (acronyms, names, etc.)
    """
    name = "hybrid_rag"

    def __init__(self, cfg: Config, collection: str = "hybrid"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._dense = VectorStore(cfg, self._embedder, collection)
        self._sparse = BM25Store(f"./data/bm25/{collection}.pkl")

    def index(self, chunks: list[Chunk]) -> None:
        self._dense.reset()
        self._sparse.reset()
        self._dense.add(chunks)
        self._sparse.add(chunks)

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        fetch = self._cfg.top_k * 3
        dense_results = self._dense.search(q_emb, fetch)
        sparse_results = self._sparse.search(question, fetch)
        fused = self._rrf([dense_results, sparse_results])[: self._cfg.top_k]
        context = self._format_context(fused)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(answer=answer, retrieved_docs=fused, queries_used=[question])

    @staticmethod
    def _rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}
        for results in lists:
            for rank, doc in enumerate(results):
                key = doc["text"]
                scores[key] += 1.0 / (k + rank + 1)
                doc_map[key] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]
