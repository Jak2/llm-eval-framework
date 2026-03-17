from __future__ import annotations

import time

import httpx

from .base import BaseLLMClient, LLMResponse

_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, prompt: str, system: str | None) -> list[dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.monotonic()
        payload = {
            "model": self._model,
            "messages": self._build_messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(_API_URL, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = int((time.monotonic() - start) * 1000)
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        result = await self.complete(prompt, system_prompt, temperature=0.0, max_tokens=max_tokens)
        return result.text
