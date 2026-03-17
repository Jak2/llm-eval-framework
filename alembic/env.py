from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_settings
from src.database.models import Base

_s = get_settings()
alembic_cfg = context.config

if alembic_cfg.config_file_name:
    fileConfig(alembic_cfg.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = create_async_engine(_s.database_url)

    async def _run() -> None:
        async with engine.connect() as conn:
            await conn.run_sync(_do_migrations)
        await engine.dispose()

    def _do_migrations(sync_conn) -> None:
        context.configure(connection=sync_conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    asyncio.run(_run())


run_migrations_online()
