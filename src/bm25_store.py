import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunker import Chunk


class BM25Store:
    """
    Sparse BM25 index. Each add() rebuilds the full index from stored documents.
    For incremental updates, existing docs from other sources are preserved.
    """

    def __init__(self, index_path: str):
        self._path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []
        if self._path.exists():
            self._load()

    def add(self, chunks: list[Chunk], *, fresh: bool = False) -> None:
        if fresh:
            self._docs = []
        else:
            sources = {c.source for c in chunks}
            self._docs = [d for d in self._docs if d["source"] not in sources]

        for c in chunks:
            self._docs.append({
                "text": c.text,
                "source": c.source,
                "chunk_id": c.chunk_id,
            })

        tokenized = [d["text"].lower().split() for d in self._docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "docs": self._docs}, f)

    def search(self, query: str, k: int) -> list[dict]:
        if not self._bm25 or not self._docs:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {**self._docs[i], "score": float(scores[i])}
            for i in top_idx
            if scores[i] > 0
        ]

    def reset(self) -> None:
        self._bm25 = None
        self._docs = []
        if self._path.exists():
            self._path.unlink()

    def _load(self) -> None:
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._docs = data["docs"]
