from __future__ import annotations

from src.config import get_settings

from .base import BaseEvaluator
from .consistency import ConsistencyChecker
from .faithfulness import FaithfulnessScorer
from .hallucination import HallucinationDetector
from .llm_judge import LLMJudgeEvaluator

_s = get_settings()

# Map evaluator name → class
_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "llm_judge": LLMJudgeEvaluator,
    "hallucination": HallucinationDetector,
    "faithfulness": FaithfulnessScorer,
    "consistency": ConsistencyChecker,
}


def get_evaluator(name: str, **overrides) -> BaseEvaluator:
    """Instantiate an evaluator by name. Keyword overrides apply to __init__."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown evaluator: {name!r}. Available: {list_evaluators()}")
    threshold = overrides.pop("threshold", _s.default_threshold)
    return cls(threshold=threshold, **overrides)


def list_evaluators() -> list[str]:
    return list(_REGISTRY)
