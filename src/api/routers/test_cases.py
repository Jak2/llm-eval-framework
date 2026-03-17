from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, update

from src.api.deps import APIKey, DBSession
from src.api.schemas import TestCaseCreate, TestCaseDetail, TestCaseRead
from src.database.models import EvalStatus, TestCase
from src.workers.tasks import run_evaluation

router = APIRouter(prefix="/test-cases", tags=["Test Cases"])


@router.post("", response_model=TestCaseRead, status_code=status.HTTP_201_CREATED)
async def create_test_case(body: TestCaseCreate, db: DBSession, _: APIKey):
    """Submit a test case and dispatch evaluation to the Celery worker."""
    tc = TestCase(**body.model_dump())
    db.add(tc)
    await db.commit()
    await db.refresh(tc)
    run_evaluation.delay(tc.id)
    return tc


@router.get("", response_model=list[TestCaseRead])
async def list_test_cases(
    db: DBSession,
    _: APIKey,
    status: str | None = None,
    llm: str | None = None,
    prompt_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    q = select(TestCase).order_by(TestCase.created_at.desc()).limit(limit).offset(offset)
    if status:
        q = q.where(TestCase.status == status)
    if llm:
        q = q.where(TestCase.llm_name == llm)
    if prompt_type:
        q = q.where(TestCase.prompt_type == prompt_type)
    return (await db.execute(q)).scalars().all()


@router.get("/{tc_id}", response_model=TestCaseDetail)
async def get_test_case(tc_id: int, db: DBSession, _: APIKey):
    tc = await db.get(TestCase, tc_id)
    if not tc:
        raise HTTPException(status_code=404, detail="Test case not found")
    return tc


@router.post("/{tc_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def rerun_test_case(tc_id: int, db: DBSession, _: APIKey):
    """Re-evaluate an existing test case (e.g. after a prompt change)."""
    tc = await db.get(TestCase, tc_id)
    if not tc:
        raise HTTPException(status_code=404, detail="Test case not found")
    await db.execute(update(TestCase).where(TestCase.id == tc_id).values(status=EvalStatus.pending))
    await db.commit()
    run_evaluation.delay(tc_id)
    return {"queued": True, "test_case_id": tc_id}


@router.delete("/{tc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_test_case(tc_id: int, db: DBSession, _: APIKey):
    tc = await db.get(TestCase, tc_id)
    if not tc:
        raise HTTPException(status_code=404, detail="Test case not found")
    await db.delete(tc)
    await db.commit()
