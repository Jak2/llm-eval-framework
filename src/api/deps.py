from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.engine import get_session

_settings = get_settings()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: Annotated[str | None, Security(_api_key_header)]) -> str:
    if key != _settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return key


# Annotated type aliases — keeps router signatures clean
DBSession = Annotated[AsyncSession, Depends(get_session)]
APIKey = Annotated[str, Depends(require_api_key)]
