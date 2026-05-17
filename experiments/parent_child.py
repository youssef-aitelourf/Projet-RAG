from src.config import Config
from src.llm import LLM
from src.embedder import Embedder
from src.vectorstore import VectorStore
from src.chunker import Chunk, chunk_text
from experiments.base import BaseRAG, RAGResult

_CHILD_SIZE = 150  # small chunks for precise embedding matching
_CHILD_OVERLAP = 20


class ParentChildRAG(BaseRAG):
    """
    Parent-Child (small-to-big) retrieval:
    - Index small child chunks (150 tokens) for precise similarity matching.
    - At query time, retrieved child IDs are mapped back to their parent chunk
      (the full ~500-token source chunk) which is passed to the LLM.
    Higher recall at retrieval (small = precise) + better context at generation (large = complete).
    """
    name = "parent_child_rag"

    def __init__(self, cfg: Config, collection: str = "parent_child"):
        self._cfg = cfg
        self._llm = LLM(cfg)
        self._embedder = Embedder(cfg)
        self._child_store = VectorStore(cfg, self._embedder, collection + "_child")
        self._parent_map: dict[str, str] = {}  # child_id → parent text

    def index(self, chunks: list[Chunk]) -> None:
        self._child_store.reset()
        self._parent_map = {}
        all_children: list[Chunk] = []

        for parent in chunks:
            children = chunk_text(
                parent.text,
                source=parent.source,
                strategy="sentence",
                chunk_size=_CHILD_SIZE,
                overlap=_CHILD_OVERLAP,
            )
            for j, child in enumerate(children):
                child_id = f"{parent.chunk_id}__c{j}"
                child.chunk_id = child_id
                self._parent_map[child_id] = parent.text
                all_children.append(child)

        self._child_store.add(all_children)

    def _run(self, question: str) -> RAGResult:
        q_emb = self._embedder.embed_one(question)
        children = self._child_store.search(q_emb, self._cfg.top_k * 2)

        # Deduplicate: map each child → parent, keep unique parents
        seen: set[str] = set()
        parent_docs: list[dict] = []
        for child in children:
            parent_text = self._parent_map.get(child["chunk_id"], child["text"])
            if parent_text not in seen:
                seen.add(parent_text)
                parent_docs.append({"text": parent_text, "source": child["source"], "score": child["score"]})
            if len(parent_docs) >= self._cfg.top_k:
                break

        context = self._format_context(parent_docs)
        answer = self._llm.ask(self._answer_prompt(question, context))
        return RAGResult(answer=answer, retrieved_docs=parent_docs, queries_used=[question])
