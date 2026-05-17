import time
from typing import Literal

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.config import Config

_MAX_RETRIES = 5
_BACKOFF_BASE = 3  # seconds

Role = Literal["generation", "eval"]


class LLM:
    def __init__(self, cfg: Config, role: Role = "generation"):
        self._cfg = cfg
        api_key, base_url, model = cfg.resolve_llm(role)
        self._model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=120.0,
        )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._cfg.temperature
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self._cfg.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except (APIConnectionError, RateLimitError) as e:
                last_exc = e
                wait = _BACKOFF_BASE * (2 ** attempt)
                print(
                    f"[llm] connection error (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
            except APIStatusError:
                raise
        raise last_exc

    def ask(self, prompt: str, temperature: float | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], temperature=temperature)
