import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.config import settings
from app.ingestion import jobs
from app.ingestion.pipeline import run_ingestion

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max is {settings.max_upload_mb}MB.",
        )

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    dest_path = str(Path(settings.upload_dir) / safe_name)
    with open(dest_path, "wb") as f:
        f.write(contents)

    job = jobs.create_job(filename=file.filename)
    background_tasks.add_task(run_ingestion, job.id, dest_path, file.filename)

    return {"job_id": job.id, "filename": file.filename, "status": job.status}


@router.get("/status/{job_id}")
async def ingest_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "error": job.error,
    }


@router.get("/jobs")
async def all_jobs():
    return [
        {
            "id": j.id,
            "filename": j.filename,
            "status": j.status,
            "total_chunks": j.total_chunks,
            "processed_chunks": j.processed_chunks,
            "error": j.error,
        }
        for j in jobs.list_jobs()
    ]
