"""Disk cache for LLM-as-judge scores to avoid redundant API calls."""

import hashlib
import json
from pathlib import Path


class JudgeCache:
    def __init__(self, cache_dir: str = "./results/.cache"):
        self._path = Path(cache_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        self._file = self._path / "judge_scores.json"
        self._data: dict[str, float] = {}
        if self._file.exists():
            try:
                self._data = json.loads(self._file.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def _key(metric: str, question: str, payload: str) -> str:
        raw = f"{metric}|{question}|{payload}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, metric: str, question: str, payload: str) -> float | None:
        return self._data.get(self._key(metric, question, payload))

    def set(self, metric: str, question: str, payload: str, score: float) -> None:
        self._data[self._key(metric, question, payload)] = score
        self._file.write_text(json.dumps(self._data, indent=2))
