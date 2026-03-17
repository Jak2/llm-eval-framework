from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    debug: bool = False
    api_key: str = "dev-secret-key"

    # LLM Providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Judge model — used by all LLM-based evaluators
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_provider: Literal["anthropic", "openai"] = "anthropic"

    # Database (asyncpg)
    database_url: str = "postgresql+asyncpg://eval_user:eval_pass@localhost:5432/llm_eval"
    db_pool_size: int = 10
    db_max_overflow: int = 5

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Evaluation
    default_threshold: float = 0.70
    consistency_runs: int = 5

    # Slack webhook for regression alerts (optional)
    slack_webhook_url: str = ""

    @property
    def judge_api_key(self) -> str:
        return self.anthropic_api_key if self.judge_provider == "anthropic" else self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
