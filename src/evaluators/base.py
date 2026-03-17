from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalResult:
    score: float        # 0.0 (fail) → 1.0 (perfect)
    passed: bool        # score >= evaluator threshold
    explanation: str    # Human-readable summary
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp to [0, 1] and round to 4 dp — guards against LLM math errors
        self.score = round(max(0.0, min(1.0, self.score)), 4)


class BaseEvaluator(ABC):
    """All evaluators extend this. Enforces a uniform interface."""

    name: str = "base"

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: str | None = None,    # RAG retrieved context
        reference: str | None = None,  # Ground truth / reference document
        **kwargs: Any,
    ) -> EvalResult:
        ...

    def _result(self, score: float, explanation: str, **metadata: Any) -> EvalResult:
        return EvalResult(
            score=score,
            passed=score >= self.threshold,
            explanation=explanation,
            metadata=metadata,
        )

    def _skip(self, reason: str) -> EvalResult:
        """Return a neutral pass when prerequisites are missing."""
        return EvalResult(score=1.0, passed=True, explanation=reason, metadata={"skipped": True})
