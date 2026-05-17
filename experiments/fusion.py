from collections import defaultdict
from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk
from experiments.base import BaseRAG, RAGResult

_EXPAND_PROMPT = """\
Generate exactly {n} alternative search queries for the question below.
Use a DIFFERENT angle for each:
  1. A definition/conceptual query ("what is X", "how does X work")
  2. A comparison/contrast query ("X vs Y", "difference between X and Y")
  3. A practical/applied query ("how to use X", "when to apply X")
Output only the queries, one per line, no numbering, no explanation.

Question: {question}

Alternative queries:"""


class RAGFusion(BaseRAG):
    """
    RAG-Fusion:
    1. Generate N alternative phrasings of the question.
    2. Retrieve top-k docs for each query variant.
    3. Merge results with Reciprocal Rank Fusion (RRF).
    4. Generate answer from fused top-k.
    Captures more diverse relevant passages than a single query.
    """
    name = "rag_fusion"

    def __init__(self, cfg: Config, collection: str = "fusion"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._store = VectorStore(cfg, self._embedder, collection)

    def index(self, chunks: list[Chunk], fresh: bool = False) -> None:
        self._prepare_vector_index(self._store, chunks, fresh)

    def _expand_queries(self, question: str) -> list[str]:
        raw = self._llm.ask(
            _EXPAND_PROMPT.format(n=self._cfg.num_fusion_queries, question=question),
            temperature=0.5,
        )
        extras = [q.strip() for q in raw.strip().splitlines() if q.strip()]
        return [question] + extras[: self._cfg.num_fusion_queries]

    @staticmethod
    def _rrf(results_per_query: list[list[dict]], k: int = 60) -> list[dict]:
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, dict] = {}
        for results in results_per_query:
            for rank, doc in enumerate(results):
                key = doc["text"]
                scores[key] += 1.0 / (k + rank + 1)
                doc_map[key] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]

    def _run(self, question: str) -> RAGResult:
        queries = self._expand_queries(question)
        all_results = [
            self._store.search(self._embedder.embed_one(q), self._cfg.top_k)
            for q in queries
        ]
        fused = self._rrf(all_results)[: self._cfg.top_k]
        context = self._format_context(fused)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(answer=answer, retrieved_docs=fused, queries_used=queries)
