from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import select, text

from src.api.deps import APIKey, DBSession
from src.api.schemas import DashboardSummary, RegressionAlert, TrendPoint
from src.database.models import EvalResult, TestCase

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def summary(db: DBSession, _: APIKey, days: int = 7):
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        await db.execute(select(EvalResult).where(EvalResult.evaluated_at >= since))
    ).scalars().all()

    if not rows:
        return DashboardSummary(
            total_runs=0, pass_rate=0.0, avg_latency_ms=0.0,
            avg_judge_score=0.0, regression_count=0,
        )

    total = len(rows)
    passed = sum(1 for r in rows if r.overall_passed)
    avg_lat = sum(r.latency_ms or 0 for r in rows) / total

    judge_scores = [
        r.eval_scores["llm_judge"]["score"]
        for r in rows
        if isinstance(r.eval_scores.get("llm_judge"), dict)
        and "score" in r.eval_scores["llm_judge"]
    ]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    return DashboardSummary(
        total_runs=total,
        pass_rate=round(passed / total, 4),
        avg_latency_ms=round(avg_lat, 1),
        avg_judge_score=round(avg_judge, 4),
        regression_count=0,  # Hydrated client-side from /regressions
    )


@router.get("/trends", response_model=list[TrendPoint])
async def trends(
    db: DBSession,
    _: APIKey,
    days: int = 7,
    llm: str | None = None,
):
    since = datetime.utcnow() - timedelta(days=days)
    q = select(EvalResult).where(EvalResult.evaluated_at >= since)
    if llm:
        q = q.join(TestCase).where(TestCase.llm_name == llm)
    rows = (await db.execute(q)).scalars().all()

    points: list[TrendPoint] = []
    for r in rows:
        date_str = r.evaluated_at.date().isoformat()
        for evaluator, data in r.eval_scores.items():
            if isinstance(data, dict) and "score" in data:
                points.append(TrendPoint(date=date_str, evaluator=evaluator, score=data["score"]))
    return points


@router.get("/regressions", response_model=list[RegressionAlert])
async def regressions(db: DBSession, _: APIKey, threshold_drop: float = 0.1):
    """Find test cases where the latest score dropped vs the previous run."""
    q = select(EvalResult).order_by(EvalResult.test_case_id, EvalResult.evaluated_at.desc())
    rows = (await db.execute(q)).scalars().all()

    # Keep last 2 results per test_case_id
    by_tc: dict[int, list[EvalResult]] = defaultdict(list)
    for r in rows:
        if len(by_tc[r.test_case_id]) < 2:
            by_tc[r.test_case_id].append(r)

    # Batch-fetch test case names
    tc_ids = list(by_tc)
    tc_map: dict[int, TestCase] = {}
    if tc_ids:
        tc_rows = (
            await db.execute(select(TestCase).where(TestCase.id.in_(tc_ids)))
        ).scalars().all()
        tc_map = {t.id: t for t in tc_rows}

    alerts: list[RegressionAlert] = []
    for tc_id, results in by_tc.items():
        if len(results) < 2:
            continue
        curr, prev = results[0], results[1]
        tc_name = tc_map[tc_id].name if tc_id in tc_map else str(tc_id)

        for evaluator in curr.eval_scores:
            prev_data = prev.eval_scores.get(evaluator, {})
            if not isinstance(prev_data, dict):
                continue
            c_score = curr.eval_scores[evaluator].get("score", 1.0)
            p_score = prev_data.get("score", 1.0)
            drop = p_score - c_score
            if drop >= threshold_drop:
                alerts.append(RegressionAlert(
                    test_case_id=tc_id,
                    test_case_name=tc_name,
                    evaluator=evaluator,
                    prev_score=round(p_score, 4),
                    curr_score=round(c_score, 4),
                    drop=round(drop, 4),
                    explanation=curr.eval_scores[evaluator].get("explanation", ""),
                ))
    return alerts


@router.get("/health")
async def health(db: DBSession):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok"}
    except Exception as exc:
        return {"status": "degraded", "database": str(exc)}
