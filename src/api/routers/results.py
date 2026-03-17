from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.api.deps import APIKey, DBSession
from src.api.schemas import EvalResultRead
from src.database.models import EvalResult

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("", response_model=list[EvalResultRead])
async def list_results(
    db: DBSession,
    _: APIKey,
    test_case_id: int | None = None,
    passed: bool | None = None,
    limit: int = 50,
    offset: int = 0,
):
    q = select(EvalResult).order_by(EvalResult.evaluated_at.desc()).limit(limit).offset(offset)
    if test_case_id is not None:
        q = q.where(EvalResult.test_case_id == test_case_id)
    if passed is not None:
        q = q.where(EvalResult.overall_passed == passed)
    return (await db.execute(q)).scalars().all()


@router.get("/{result_id}", response_model=EvalResultRead)
async def get_result(result_id: int, db: DBSession, _: APIKey):
    r = await db.get(EvalResult, result_id)
    if not r:
        raise HTTPException(status_code=404, detail="Result not found")
    return r
