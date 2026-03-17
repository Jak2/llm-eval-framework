from __future__ import annotations

import asyncio
import logging
from itertools import combinations
from typing import Any

from src.llm_clients.base import BaseLLMClient
from src.llm_clients.registry import get_llm_client

from .base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard — O(n) lightweight fallback similarity."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine_dot(a: list[float], b: list[float]) -> float:
    """Dot product of two unit-normalised vectors (= cosine similarity)."""
    return sum(x * y for x, y in zip(a, b))


class ConsistencyChecker(BaseEvaluator):
    """Measures output stability by running the same prompt N times.

    Similarity strategy (in priority order):
      1. sentence-transformers cosine similarity if installed (~80 MB, best quality)
      2. Token Jaccard (zero dependencies, reasonable proxy)

    Score = mean pairwise similarity. Ideal for factual prompts: > 0.90.
    """

    name = "consistency"

    def __init__(
        self,
        n_runs: int = 5,
        threshold: float = 0.85,
        llm_name: str = "claude",
    ) -> None:
        super().__init__(threshold)
        self.n_runs = n_runs
        self.llm_name = llm_name
        self._embedder = None   # Lazy-loaded — avoids 80 MB import on startup

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("sentence-transformers loaded for consistency scoring")
        except ImportError:
            logger.info("sentence-transformers not installed — using Jaccard similarity")
        return self._embedder

    def _similarities(self, texts: list[str]) -> list[float]:
        embedder = self._get_embedder()
        if embedder is not None:
            import numpy as np  # noqa: PLC0415
            embs = embedder.encode(texts, normalize_embeddings=True)
            pairs = list(combinations(range(len(embs)), 2))
            return [float(np.dot(embs[i], embs[j])) for i, j in pairs]
        # Fallback
        return [_jaccard(a, b) for a, b in combinations(texts, 2)]

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
        reference: str | None = None,
        llm_client: BaseLLMClient | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        client = llm_client or get_llm_client(self.llm_name)

        # Run N-1 extra completions concurrently alongside the original response
        extra_tasks = [client.complete(prompt, temperature=0.7) for _ in range(self.n_runs - 1)]
        extra_results = await asyncio.gather(*extra_tasks, return_exceptions=True)

        responses = [response]
        for r in extra_results:
            if isinstance(r, Exception):
                logger.warning("Consistency re-run failed: %s", r)
            else:
                responses.append(r.text)

        if len(responses) < 2:
            return self._result(0.5, "Too few successful runs for consistency check.")

        sims = self._similarities(responses)
        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        std_sim = variance ** 0.5

        label = "STABLE" if mean_sim >= self.threshold else "INCONSISTENT"
        explanation = (
            f"Mean similarity: {mean_sim:.3f} (std: {std_sim:.3f}) "
            f"over {len(responses)} runs — {label}."
        )
        return self._result(mean_sim, explanation, std=round(std_sim, 4), n_runs=len(responses))
