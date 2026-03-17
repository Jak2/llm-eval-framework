from __future__ import annotations

import json
import logging
from typing import Any

from src.llm_clients.base import BaseLLMClient
from src.llm_clients.registry import get_judge_client

from .base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a hallucination auditor. Check every factual claim in the AI response against the reference text.

A claim is HALLUCINATED if it:
  (a) Cannot be inferred from the reference text, OR
  (b) Contradicts the reference text, OR
  (c) States a specific fact (date, number, name, quote) not present in the reference

Return ONLY valid JSON (no markdown, no backticks):
{
  "claims": [
    {
      "claim": "<exact text of the claim>",
      "verdict": "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
      "evidence": "<direct quote from reference that supports/contradicts, or null>"
    }
  ],
  "hallucination_rate": <float 0.0-1.0: proportion of non-SUPPORTED claims>,
  "faithfulness_score": <float: 1.0 - hallucination_rate>
}"""


class HallucinationDetector(BaseEvaluator):
    """LLM-based NLI — checks every claim against reference text.

    No local model required. Uses the judge LLM (Claude/GPT-4) via API.
    For local NLI using cross-encoder/nli-deberta-v3-base, set use_local_nli=True
    in the evaluator registry (requires transformers + torch to be installed).
    """

    name = "hallucination"

    def __init__(self, threshold: float = 0.75, client: BaseLLMClient | None = None) -> None:
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
        source = reference or context
        if not source:
            return self._skip("No reference text — hallucination check skipped.")

        user_msg = (
            f"REFERENCE TEXT:\n{source[:3000]}\n\n"
            f"AI RESPONSE TO EVALUATE:\n{response}"
        )
        try:
            raw = await self._client.complete_json(user_msg, _SYSTEM_PROMPT, max_tokens=1500)
            data = _parse_json(raw)
        except Exception as exc:
            logger.warning("hallucination parse error: %s", exc)
            return self._result(0.5, f"Parse error: {exc}")

        score = float(data.get("faithfulness_score", 0.5))
        claims = data.get("claims", [])
        bad = [c for c in claims if c.get("verdict") != "SUPPORTED"]
        supported = len(claims) - len(bad)

        explanation = (
            f"{supported}/{len(claims)} claims supported. "
            + (f"Flagged: \"{bad[0]['claim'][:80]}\"" if bad else "All claims verified.")
        )
        return self._result(
            score,
            explanation,
            claims=claims,
            hallucination_rate=data.get("hallucination_rate", 0.0),
        )


def _parse_json(raw: str) -> dict:
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)
