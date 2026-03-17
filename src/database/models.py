from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EvalStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TestCase(Base):
    __tablename__ = "test_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    context: Mapped[str | None] = mapped_column(Text)           # RAG retrieved context
    reference_answer: Mapped[str | None] = mapped_column(Text)  # Ground truth for hallucination
    llm_name: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_config: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluators: Mapped[list] = mapped_column(JSON, default=list)
    prompt_type: Mapped[str] = mapped_column(String(100), default="general")
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(Enum(EvalStatus), default=EvalStatus.pending)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    results: Mapped[list[EvalResult]] = relationship(
        "EvalResult",
        back_populates="test_case",
        lazy="select",
        cascade="all, delete-orphan",
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("test_cases.id", ondelete="CASCADE"), index=True
    )
    llm_response: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    # {"llm_judge": {"score": 0.85, "passed": true, "explanation": "...", "metadata": {...}}, ...}
    eval_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    overall_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    run_number: Mapped[int] = mapped_column(Integer, default=1)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    test_case: Mapped[TestCase] = relationship("TestCase", back_populates="results")
