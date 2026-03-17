from __future__ import annotations

from src.config import get_settings

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .openai_client import OpenAIClient

_s = get_settings()


def get_judge_client() -> BaseLLMClient:
    """Returns the configured judge LLM (used by all LLM-based evaluators)."""
    if _s.judge_provider == "openai":
        return OpenAIClient(api_key=_s.openai_api_key, model=_s.judge_model)
    return AnthropicClient(api_key=_s.anthropic_api_key, model=_s.judge_model)


def get_llm_client(name: str, config: dict | None = None) -> BaseLLMClient:
    """Factory — resolves an LLM client by name with optional per-request config."""
    cfg = config or {}
    match name.lower():
        case "claude" | "anthropic":
            return AnthropicClient(
                api_key=cfg.get("api_key", _s.anthropic_api_key),
                model=cfg.get("model", "claude-haiku-4-5-20251001"),
            )
        case "openai" | "gpt4" | "gpt-4" | "gpt-4o" | "gpt-4o-mini":
            return OpenAIClient(
                api_key=cfg.get("api_key", _s.openai_api_key),
                model=cfg.get("model", "gpt-4o-mini"),
            )
        case _:
            raise ValueError(
                f"Unknown LLM provider: {name!r}. Supported: 'claude', 'openai'"
            )
