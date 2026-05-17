import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import Config


class Embedder:
    def __init__(self, cfg: Config):
        self._model = SentenceTransformer(cfg.embedding_model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
