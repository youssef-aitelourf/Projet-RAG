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


class CompressionRAG(BaseRAG):
    """
    Contextual Compression RAG:
    1. Retrieve top_k * 2 candidate chunks (larger candidate pool).
    2. For each candidate, ask the LLM to extract only relevant sentences.
    3. Discard chunks marked IRRELEVANT.
    4. Generate from the compressed, noise-free context.
    Reduces distracting content in the context window.
    """
    name = "compression_rag"

    def __init__(self, cfg: Config, collection: str = "compression"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)

    def index(self, chunks: list[Chunk]) -> None:
        self._store.reset()
        self._store.add(chunks)

    def _compress(self, question: str, doc: dict) -> dict | None:
        compressed = self._llm.ask(
            _COMPRESS_PROMPT.format(question=question, passage=doc["text"]),
            temperature=0.0,
        )
        if compressed.strip().upper().startswith("IRRELEVANT"):
            return None
        return {**doc, "text": compressed.strip()}

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        candidates = self._store.search(q_emb, self._cfg.top_k * 2)

        compressed_docs: list[dict] = []
        for doc in candidates:
            result = self._compress(question, doc)
            if result:
                compressed_docs.append(result)
            if len(compressed_docs) >= self._cfg.top_k:
                break

        # Fallback: if all compressed away, use raw top-k
        final_docs = compressed_docs if compressed_docs else candidates[: self._cfg.top_k]
        context = self._format_context(final_docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(
            answer=answer,
            retrieved_docs=final_docs,
            queries_used=[question],
        )
