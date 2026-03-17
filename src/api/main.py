from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.database.engine import init_db

from .routers import dashboard, results, test_cases

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LLM Eval Framework starting — creating tables if needed")
    await init_db()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="LLM Evaluation Framework",
    version="1.0.0",
    description=(
        "Production-grade automated LLM quality testing. "
        "Evaluates: LLM-as-judge, hallucination, faithfulness, consistency."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers under /api
for router in (test_cases.router, results.router, dashboard.router):
    app.include_router(router, prefix="/api")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}
