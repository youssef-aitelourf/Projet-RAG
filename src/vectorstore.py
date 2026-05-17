import chromadb
import numpy as np
from src.chunker import Chunk
from src.embedder import Embedder
from src.config import Config


class VectorStore:
    def __init__(self, cfg: Config, embedder: Embedder, collection_name: str = "rag"):
        self._cfg = cfg
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=cfg.chroma_dir)
        self._name = collection_name
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.embed([c.text for c in chunks]).tolist()
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source} for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, k: int) -> list[dict]:
        k = min(k, self._col.count())
        if k == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )
        docs = []
        for i, text in enumerate(res["documents"][0]):
            docs.append({
                "text": text,
                "source": res["metadatas"][0][i]["source"],
                "score": 1.0 - res["distances"][0][i],
                "chunk_id": res["ids"][0][i],
            })
        return docs

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._col.count()
