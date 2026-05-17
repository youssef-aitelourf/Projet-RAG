import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("ANTHORPIC_API_KEY", "")


@dataclass
class Config:
    # LLM provider: ollama_cloud | ollama_local | openai | anthropic
    provider: str = field(
        default_factory=lambda: os.getenv("RAG_PROVIDER", "anthropic").lower()
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_MODEL", "claude-sonnet-4-20250514"
        )
    )
    eval_model: str = field(default_factory=lambda: os.getenv("EVAL_MODEL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=_anthropic_api_key)
    temperature: float = 0.0
    max_tokens: int = 1024

    embedding_model: str = "all-MiniLM-L6-v2"

    top_k: int = 5
    top_k_fetch: int = 20
    num_fusion_queries: int = 3

    compression_mode: str = field(
        default_factory=lambda: os.getenv("COMPRESSION_MODE", "extractive").lower()
    )

    chunk_size: int = 500
    chunk_overlap: int = 50

    chroma_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_DIR", "./data/chroma")
    )
    results_dir: str = "./results"

    @property
    def llm_backend(self) -> str:
        """openai_compatible | anthropic"""
        if self.provider == "anthropic":
            return "anthropic"
        return "openai_compatible"

    def resolve_llm(self, role: str = "generation") -> tuple[str, str, str]:
        """Return (api_key, base_url, model). base_url empty for Anthropic."""
        model = self.eval_model if role == "eval" and self.eval_model else self.model

        if self.provider == "anthropic":
            key = self.anthropic_api_key
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when RAG_PROVIDER=anthropic. "
                    "Set it in .env (check spelling: ANTHROPIC not ANTHORPIC)."
                )
            return key, "", model

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
                    "OPENAI_API_KEY is required when RAG_PROVIDER=openai."
                )
            return (
                key,
                self.base_url or "https://api.openai.com/v1",
                model,
            )

        key = self.api_key
        if not key:
            raise ValueError(
                "OLLAMA_API_KEY is required when RAG_PROVIDER=ollama_cloud."
            )
        return (
            key,
            self.base_url or "https://ollama.com/v1",
            model,
        )
