from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult


class NaiveRAG(BaseRAG):
    """
    Baseline RAG: embed query → cosine search → top-k → generate.
    No query transformation, no reranking.
    """
    name = "naive_rag"

    def __init__(self, cfg: Config, collection: str = "naive"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)

    def index(self, chunks: list[Chunk]) -> None:
        self._store.reset()
        self._store.add(chunks)

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        docs = self._store.search(q_emb, self._cfg.top_k)
        context = self._format_context(docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(answer=answer, retrieved_docs=docs, queries_used=[question])
