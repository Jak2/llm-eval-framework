from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.evaluators.base import EvalResult
from src.evaluators.consistency import ConsistencyChecker, _jaccard
from src.evaluators.faithfulness import FaithfulnessScorer
from src.evaluators.hallucination import HallucinationDetector
from src.evaluators.llm_judge import LLMJudgeEvaluator, _parse_json
from src.evaluators.registry import get_evaluator, list_evaluators


# ── EvalResult ────────────────────────────────────────────────────────────────

def test_eval_result_clamps_score_above_1():
    r = EvalResult(score=1.5, passed=True, explanation="test")
    assert r.score == 1.0


def test_eval_result_clamps_score_below_0():
    r = EvalResult(score=-0.3, passed=False, explanation="test")
    assert r.score == 0.0


def test_eval_result_rounds_to_4dp():
    r = EvalResult(score=0.666666, passed=True, explanation="test")
    assert r.score == 0.6667


# ── LLM Judge ─────────────────────────────────────────────────────────────────

def _mock_client(json_response: str) -> MagicMock:
    client = MagicMock()
    client.complete_json = AsyncMock(return_value=json_response)
    return client


@pytest.mark.asyncio
async def test_llm_judge_passes_on_perfect_scores():
    client = _mock_client("""{
        "accuracy": 5, "relevance": 5, "completeness": 5,
        "clarity": 5, "safety": 5, "overall": 5.0,
        "reasoning": "Perfect.", "critical_issue": null,
        "recommendation": "PASS"
    }""")
    evaluator = LLMJudgeEvaluator(threshold=0.6, client=client)
    result = await evaluator.evaluate("What is 2+2?", "4")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_llm_judge_fails_on_low_scores():
    client = _mock_client("""{
        "accuracy": 1, "relevance": 1, "completeness": 1,
        "clarity": 1, "safety": 5, "overall": 1.8,
        "reasoning": "Terrible.", "critical_issue": "Wrong",
        "recommendation": "FAIL"
    }""")
    evaluator = LLMJudgeEvaluator(threshold=0.6, client=client)
    result = await evaluator.evaluate("Q", "garbage answer")
    assert result.passed is False
    assert result.score < 0.3


def test_parse_json_strips_markdown_fences():
    raw = "```json\n{\"accuracy\": 5, \"relevance\": 4, \"completeness\": 3, \"clarity\": 4, \"safety\": 5, \"overall\": 4.2, \"reasoning\": \"ok\", \"critical_issue\": null, \"recommendation\": \"PASS\"}\n```"
    data = _parse_json(raw)
    assert data["accuracy"] == 5


# ── Hallucination ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hallucination_skips_when_no_reference():
    evaluator = HallucinationDetector(client=_mock_client("{}"))
    result = await evaluator.evaluate("Q", "A")
    assert result.score == 1.0
    assert result.metadata.get("skipped") is True


@pytest.mark.asyncio
async def test_hallucination_scores_all_supported():
    client = _mock_client('{"claims": [{"claim": "Paris is the capital.", "verdict": "SUPPORTED", "evidence": "Paris is the capital of France."}], "hallucination_rate": 0.0, "faithfulness_score": 1.0}')
    evaluator = HallucinationDetector(threshold=0.75, client=client)
    result = await evaluator.evaluate("Q", "Paris is the capital.", reference="Paris is the capital of France.")
    assert result.score == 1.0
    assert result.passed is True


# ── Faithfulness ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_faithfulness_skips_when_no_context():
    evaluator = FaithfulnessScorer(client=_mock_client("{}"))
    result = await evaluator.evaluate("Q", "A")
    assert result.score == 1.0
    assert result.metadata.get("skipped") is True


# ── Consistency ───────────────────────────────────────────────────────────────

def test_jaccard_identical_strings():
    assert _jaccard("the cat sat", "the cat sat") == 1.0


def test_jaccard_disjoint_strings():
    assert _jaccard("foo bar", "baz qux") == 0.0


@pytest.mark.asyncio
async def test_consistency_handles_failed_reruns():
    """If extra runs fail, consistency should still return a result."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=Exception("API rate limit"))
    evaluator = ConsistencyChecker(n_runs=3, threshold=0.85, llm_name="claude")
    # Only original response, all re-runs fail
    result = await evaluator.evaluate("Q", "A", llm_client=mock_llm)
    assert result.score >= 0.0  # Graceful degradation


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_lists_all_evaluators():
    names = list_evaluators()
    assert "llm_judge" in names
    assert "hallucination" in names
    assert "faithfulness" in names
    assert "consistency" in names


def test_registry_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("does_not_exist")
