from __future__ import annotations

import json
import logging
from typing import Any

from src.llm_clients.base import BaseLLMClient
from src.llm_clients.registry import get_judge_client

from .base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You evaluate a RAG system's faithfulness: does the generated answer stay within the retrieved context?

A statement is UNSUPPORTED if it introduces knowledge not present in the retrieved context — even if
that knowledge is factually correct. This catches "context leakage" where the model ignores the
retriever and uses its parametric knowledge instead.

Return ONLY valid JSON (no markdown, no backticks):
{
  "statements": [
    {
      "text": "<claim from the answer>",
      "classification": "SUPPORTED" | "UNSUPPORTED",
      "evidence": "<exact quote from context that supports it, or null>"
    }
  ],
  "faithfulness_score": <float 0.0-1.0: proportion SUPPORTED>,
  "unsupported_count": <int>
}"""


class FaithfulnessScorer(BaseEvaluator):
    """RAG-specific: measures if the answer is grounded in retrieved context.

    Faithfulness ≠ accuracy. A score of 1.0 means the model only used retrieved
    context — even if that context was wrong. Use HallucinationDetector for
    factual accuracy against a known reference.
    """

    name = "faithfulness"

    def __init__(self, threshold: float = 0.8, client: BaseLLMClient | None = None) -> None:
        super().__init__(threshold)
        self._client = client or get_judge_client()

    async def evaluate(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
        reference: str | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        if not context:
            return self._skip("No RAG context provided — faithfulness check skipped.")

        user_msg = (
            f"RETRIEVED CONTEXT:\n{context[:2000]}\n\n"
            f"GENERATED ANSWER:\n{response}"
        )
        try:
            raw = await self._client.complete_json(user_msg, _SYSTEM_PROMPT, max_tokens=1500)
            text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(text)
        except Exception as exc:
            logger.warning("faithfulness parse error: %s", exc)
            return self._result(0.5, f"Parse error: {exc}")

        score = float(data.get("faithfulness_score", 0.5))
        unsupported = int(data.get("unsupported_count", 0))
        statements = data.get("statements", [])

        explanation = (
            f"Faithfulness: {score:.0%}. "
            f"{unsupported} unsupported statement(s) out of {len(statements)}."
        )
        return self._result(score, explanation, statements=statements)
