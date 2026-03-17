from __future__ import annotations

import time

import httpx

from .base import BaseLLMClient, LLMResponse

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

    async def _post(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        resp = await client.post(_API_URL, headers=self._headers(), json=payload)
        resp.raise_for_status()
        return resp.json()

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            data = await self._post(
                client,
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        latency = int((time.monotonic() - start) * 1000)
        usage = data.get("usage", {})
        return LLMResponse(
            text=data["content"][0]["text"],
            model=data.get("model", self._model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=latency,
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            data = await self._post(
                client,
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt,
                temperature=0.0,    # Deterministic for evaluation
                max_tokens=max_tokens,
            )
        return data["content"][0]["text"]
