from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Test Case ─────────────────────────────────────────────────────────────────

class TestCaseCreate(BaseModel):
    name: str = Field(..., max_length=255)
    prompt: str
    system_prompt: str | None = None
    context: str | None = None          # RAG retrieved context
    reference_answer: str | None = None # Ground truth for hallucination
    llm_name: str = Field(..., examples=["claude", "openai"])
    llm_config: dict[str, Any] = Field(default_factory=dict)
    evaluators: list[str] = Field(
        default=["llm_judge"],
        examples=[["llm_judge", "hallucination", "faithfulness"]],
    )
    prompt_type: str = Field("general", max_length=100)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class TestCaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    prompt: str
    llm_name: str
    evaluators: list[str]
    prompt_type: str
    temperature: float
    status: str
    created_at: datetime


class TestCaseDetail(TestCaseRead):
    system_prompt: str | None
    context: str | None
    reference_answer: str | None
    llm_config: dict[str, Any]


# ── Eval Result ───────────────────────────────────────────────────────────────

class EvalResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    test_case_id: int
    llm_response: str
    latency_ms: int | None
    token_count: int | None
    eval_scores: dict[str, Any]
    overall_passed: bool
    run_number: int
    evaluated_at: datetime


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    total_runs: int
    pass_rate: float
    avg_latency_ms: float
    avg_judge_score: float
    regression_count: int


class TrendPoint(BaseModel):
    date: str
    evaluator: str
    score: float


class RegressionAlert(BaseModel):
    test_case_id: int
    test_case_name: str
    evaluator: str
    prev_score: float
    curr_score: float
    drop: float
    explanation: str
