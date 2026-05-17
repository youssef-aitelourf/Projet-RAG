import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.config import Config

_MAX_RETRIES = 5
_BACKOFF_BASE = 3  # seconds


class LLM:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            max_retries=0,  # we handle retries ourselves
            timeout=120.0,
        )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._cfg.temperature
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self._cfg.max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except (APIConnectionError, RateLimitError) as e:
                last_exc = e
                wait = _BACKOFF_BASE * (2 ** attempt)
                print(f"[llm] connection error (attempt {attempt+1}/{_MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                raise  # non-retryable (4xx / 5xx from server)
        raise last_exc  # exhausted retries

    def ask(self, prompt: str, temperature: float | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], temperature=temperature)
