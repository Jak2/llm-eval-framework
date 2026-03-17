from celery import Celery

from src.config import get_settings

_s = get_settings()

celery_app = Celery(
    "llm_eval",
    broker=_s.celery_broker_url,
    backend=_s.celery_result_backend,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,           # Ack only after task completes — prevents data loss on crash
    worker_prefetch_multiplier=1,  # One task per worker — fair dispatch for long eval jobs
    result_expires=86_400,         # 24 h
    task_soft_time_limit=300,      # 5 min soft limit per task
    task_time_limit=360,           # 6 min hard limit
)
