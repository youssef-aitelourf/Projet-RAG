import time
from typing import Literal

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.config import Config

_MAX_RETRIES = 5
_BACKOFF_BASE = 3

Role = Literal["generation", "eval"]


class LLM:
    def __init__(self, cfg: Config, role: Role = "generation"):
        self._cfg = cfg
        api_key, base_url, model = cfg.resolve_llm(role)
        self._model = model
        self._backend = cfg.llm_backend

        if self._backend == "anthropic":
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=api_key)
            self._client = None
        else:
            self._anthropic = None
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                timeout=120.0,
            )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._cfg.temperature
        if self._backend == "anthropic":
            return self._chat_anthropic(messages, temp)
        return self._chat_openai(messages, temp)

    def _chat_anthropic(self, messages: list[dict], temperature: float) -> str:
        system_parts = [
            m["content"] for m in messages if m.get("role") == "system"
        ]
        user_parts = [
            m["content"] for m in messages if m.get("role") == "user"
        ]
        prompt = "\n\n".join(user_parts) if user_parts else messages[-1]["content"]
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._cfg.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._anthropic.messages.create(**kwargs)
                return resp.content[0].text.strip()
            except Exception as e:
                last_exc = e
                err = str(e).lower()
                if "rate" in err or "overloaded" in err or "timeout" in err:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    print(
                        f"[llm] anthropic retry ({attempt + 1}/{_MAX_RETRIES}) "
                        f"in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _chat_openai(self, messages: list[dict], temperature: float) -> str:
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
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
