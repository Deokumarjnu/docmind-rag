"""Celery application configuration."""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "docmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout for large documents
    worker_prefetch_multiplier=1,  # Process one task at a time
)

