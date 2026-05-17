import os
from dataclasses import dataclass, field

Provider = str  # ollama_cloud | ollama_local | openai


@dataclass
class Config:
    # LLM provider: ollama_cloud (default), ollama_local, openai
    provider: str = field(
        default_factory=lambda: os.getenv("RAG_PROVIDER", "ollama_cloud").lower()
    )
    model: str = field(default_factory=lambda: os.getenv("RAG_MODEL", "gpt-oss:120b"))
    eval_model: str = field(default_factory=lambda: os.getenv("EVAL_MODEL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    temperature: float = 0.0
    max_tokens: int = 1024

    # Local embeddings (sentence-transformers, no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    top_k: int = 5
    top_k_fetch: int = 20  # for reranker: fetch more, then rerank down to top_k

    # RAG-Fusion
    num_fusion_queries: int = 3

    # Compression: llm | extractive
    compression_mode: str = field(
        default_factory=lambda: os.getenv("COMPRESSION_MODE", "extractive").lower()
    )

    # Chunking defaults
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Paths
    chroma_dir: str = "./data/chroma"
    results_dir: str = "./results"

    def resolve_llm(self, role: str = "generation") -> tuple[str, str, str]:
        """Return (api_key, base_url, model) for the given role."""
        model = self.eval_model if role == "eval" and self.eval_model else self.model

        if self.provider == "ollama_local":
            return (
                self.api_key or "ollama",
                self.base_url or "http://localhost:11434/v1",
                model,
            )
        if self.provider == "openai":
            key = self.openai_api_key
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY is required when RAG_PROVIDER=openai. "
                    "Set it in .env or switch RAG_PROVIDER."
                )
            return (
                key,
                self.base_url or "https://api.openai.com/v1",
                model,
            )
        # ollama_cloud (default)
        key = self.api_key
        if not key:
            raise ValueError(
                "OLLAMA_API_KEY is required when RAG_PROVIDER=ollama_cloud. "
                "Get a key at https://ollama.com/settings/keys"
            )
        return (
            key,
            self.base_url or "https://ollama.com/v1",
            model,
        )
