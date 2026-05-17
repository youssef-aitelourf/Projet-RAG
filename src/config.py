import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Ollama Cloud LLM
    model: str = field(default_factory=lambda: os.getenv("RAG_MODEL", "gpt-oss:120b"))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"))
    temperature: float = 0.0
    max_tokens: int = 1024

    # Local embeddings (sentence-transformers, no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5
    top_k_fetch: int = 20  # for reranker: fetch more, then rerank down to top_k

    # RAG-Fusion
    num_fusion_queries: int = 3

    # Chunking defaults
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Paths
    chroma_dir: str = "./data/chroma"
    results_dir: str = "./results"
