from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult

_HYPOTHESIS_PROMPT = """\
Write a short, factual passage that would directly answer the question below.
Be specific and confident — do not say you don't know.

Question: {question}

Hypothetical passage:"""


class HyDERAG(BaseRAG):
    """
    HyDE (Hypothetical Document Embeddings):
    Instead of embedding the raw question, ask the LLM to generate a
    hypothetical answer, then embed THAT to retrieve similar real documents.
    Better for abstract or knowledge-dense queries.
    """
    name = "hyde_rag"

    def __init__(self, cfg: Config, collection: str = "hyde"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)

    def index(self, chunks: list[Chunk], fresh: bool = False) -> None:
        self._prepare_vector_index(self._store, chunks, fresh)

    def _run(self, question: str) -> RAGResult:
        hypothesis = self._llm.ask(_HYPOTHESIS_PROMPT.format(question=question), temperature=0.3)
        h_emb = self._embedder.embed_one(hypothesis)
        docs = self._store.search(h_emb, self._cfg.top_k)
        context = self._format_context(docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            queries_used=[question, f"[hypothesis] {hypothesis[:100]}..."],
        )
