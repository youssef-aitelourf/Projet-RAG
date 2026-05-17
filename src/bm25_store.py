import pickle
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunker import Chunk


class BM25Store:
    def __init__(self, index_path: str):
        self._path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []
        if self._path.exists():
            self._load()

    def add(self, chunks: list[Chunk]) -> None:
        self._docs = [{"text": c.text, "source": c.source} for c in chunks]
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
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
