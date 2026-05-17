import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.chunker import Chunk


@dataclass
class RAGResult:
    answer: str
    retrieved_docs: list[dict]
    queries_used: list[str]
    latency: float = 0.0


class BaseRAG(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def _run(self, question: str) -> RAGResult: ...

    def query(self, question: str) -> RAGResult:
        t0 = time.perf_counter()
        result = self._run(question)
        result.latency = time.perf_counter() - t0
        return result

    @staticmethod
    def _format_context(docs: list[dict]) -> str:
        return "\n\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))

    @staticmethod
    def _answer_prompt(question: str, context: str) -> str:
        return (
            "Answer the question using ONLY the context below. "
            "If the context lacks the information, say so explicitly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
