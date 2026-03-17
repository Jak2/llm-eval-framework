from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLMClient(ABC):
    """Uniform interface for all LLM providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a prompt and return the full response."""
        ...

    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Complete and return raw text expected to be valid JSON.
        Callers are responsible for json.loads()."""
        ...
