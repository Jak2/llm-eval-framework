from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime

import httpx

from src.config import get_settings
from src.database.engine import AsyncSessionLocal
from src.database.models import EvalResult, EvalStatus, TestCase
from src.evaluators.registry import get_evaluator
from src.llm_clients.registry import get_llm_client
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
_s = get_settings()


# ── Public Celery Task ────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def run_evaluation(self, test_case_id: int) -> dict:
    """Entry point dispatched by the FastAPI layer. Wraps the async pipeline."""
    try:
        return asyncio.run(_evaluate(test_case_id))
    except Exception as exc:
        logger.error("Evaluation failed for tc=%d: %s", test_case_id, exc)
        raise self.retry(exc=exc)


# ── Async Pipeline ────────────────────────────────────────────────────────────

async def _evaluate(test_case_id: int) -> dict:
    async with AsyncSessionLocal() as db:
        tc = await db.get(TestCase, test_case_id)
        if not tc:
            raise ValueError(f"TestCase {test_case_id} not found")

        tc.status = EvalStatus.running
        await db.commit()

        try:
            result = await _run_evals(tc)
            db.add(result)
            tc.status = EvalStatus.completed
            await db.commit()
            await db.refresh(result)

            if _s.slack_webhook_url and not result.overall_passed:
                await _slack_alert(result)

            return {"status": "completed", "result_id": result.id}

        except Exception as exc:
            tc.status = EvalStatus.failed
            await db.commit()
            raise


async def _run_evals(tc: TestCase) -> EvalResult:
    """Calls the target LLM and runs every configured evaluator."""
    llm_client = get_llm_client(tc.llm_name, tc.llm_config)

    start = time.monotonic()
    llm_resp = await llm_client.complete(
        tc.prompt,
        system_prompt=tc.system_prompt,
        temperature=tc.temperature,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    eval_scores: dict = {}
    for name in tc.evaluators:
        try:
            evaluator = get_evaluator(name)
            result = await evaluator.evaluate(
                prompt=tc.prompt,
                response=llm_resp.text,
                context=tc.context,
                reference=tc.reference_answer,
            )
            eval_scores[name] = {
                "score": result.score,
                "passed": result.passed,
                "explanation": result.explanation,
                "metadata": result.metadata,
            }
        except Exception as exc:
            logger.warning("Evaluator %r failed: %s", name, exc)
            eval_scores[name] = {
                "score": 0.0,
                "passed": False,
                "explanation": f"Evaluator error: {exc}",
                "metadata": {},
            }

    overall_passed = all(v["passed"] for v in eval_scores.values())
    return EvalResult(
        test_case_id=tc.id,
        llm_response=llm_resp.text,
        latency_ms=latency_ms,
        token_count=llm_resp.total_tokens,
        eval_scores=eval_scores,
        overall_passed=overall_passed,
        evaluated_at=datetime.utcnow(),
    )


async def _slack_alert(result: EvalResult) -> None:
    score_lines = "\n".join(
        f"  • {k}: {v['score']:.2f} ({'✅' if v['passed'] else '❌'})"
        for k, v in result.eval_scores.items()
    )
    payload = {
        "text": (
            f"*LLM Eval Alert* — TestCase #{result.test_case_id} *FAILED*\n"
            f"{score_lines}"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(_s.slack_webhook_url, json=payload)
    except Exception as exc:
        logger.warning("Slack alert failed: %s", exc)
