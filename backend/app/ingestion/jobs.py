"""
In-memory ingestion job tracker.

This is a single-process, in-memory store -- perfectly fine for the local,
single-user/demo deployment this project targets (docker compose on one
machine). If you outgrow that (multiple backend replicas, need for job
history across restarts), swap this for a Redis- or Postgres-backed store;
the rest of the app only depends on the small interface below.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

JobStatus = Literal["pending", "processing", "done", "error"]
JobPhase = Literal["embed", "extract"]


@dataclass
class IngestJob:
    id: str
    filename: str
    status: JobStatus = "pending"
    phase: JobPhase | None = None
    total_chunks: int = 0
    processed_chunks: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_jobs: dict[str, IngestJob] = {}


def create_job(filename: str) -> IngestJob:
    job = IngestJob(id=str(uuid.uuid4()), filename=filename)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> IngestJob | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    for key, value in kwargs.items():
        setattr(job, key, value)
    job.updated_at = time.time()


def list_jobs() -> list[IngestJob]:
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
