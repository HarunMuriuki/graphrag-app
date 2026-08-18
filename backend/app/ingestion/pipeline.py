"""
End-to-end ingestion pipeline, run as a FastAPI BackgroundTask:

    load file -> extract text -> chunk -> for each chunk (in parallel):
        embed (Ollama/nomic-embed-text) -> upsert into Qdrant
        extract entities/relations (Ollama/mistral) -> write into Neo4j

Every chunk gets a stable UUID (derived from doc_id + chunk_index) that is
shared between Qdrant (point id) and Neo4j (Chunk.id), which is what lets
retrieval hop from "these chunks matched the vector search" to "these
entities/relationships are attached to those chunks" in the graph.

PARALLELISM:
    Chunks are processed concurrently up to `settings.ingestion_concurrency`
    (default 4). This bounds how many Ollama requests are in-flight at once
    so we don't overwhelm GPU memory. The semaphore is acquired BEFORE the
    Ollama calls (the slow part) and released AFTER the DB writes (the fast
    part), keeping each chunk's work atomic.

    Progress tracking uses an atomic counter (completed_count + Lock) instead
    of sequential index, because chunks now finish out of order.

    To disable parallelism, set INGESTION_CONCURRENCY=1 in your .env.
"""
import asyncio
import logging
import uuid
from pathlib import Path

from app.core.config import settings
from app.db import neo4j_client, qdrant_client
from app.db.ollama_client import ollama_client
from app.ingestion import loaders
from app.ingestion.chunker import chunk_text
from app.ingestion.extractor import extract_entities_relations
from app.ingestion.jobs import update_job

logger = logging.getLogger("graphrag.ingestion")


async def run_ingestion(job_id: str, file_path: str, original_filename: str) -> None:
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_filename))
    try:
        update_job(job_id, status="processing")

        text = loaders.extract_text(file_path)
        chunks = chunk_text(text)
        update_job(job_id, total_chunks=len(chunks))

        if not chunks:
            update_job(job_id, status="error", error="No extractable text found in file.")
            return

        # --- Parallel chunk processing ---
        # The semaphore caps how many chunks are processed at the same time.
        # This prevents Ollama from running out of GPU memory when ingesting
        # large documents with many chunks. Increase if your Ollama server
        # has enough VRAM (e.g. 8+ GB), decrease if you see OOM errors.
        semaphore = asyncio.Semaphore(settings.ingestion_concurrency)
        completed_count = 0
        progress_lock = asyncio.Lock()

        async def _process_chunk(index: int, chunk: str) -> None:
            nonlocal completed_count

            # Acquire semaphore BEFORE any Ollama call — this is the
            # bottleneck that needs rate limiting, not the DB writes.
            async with semaphore:
                chunk_id = qdrant_client.chunk_point_id(doc_id, index)

                # Embedding is fast (~100-300ms) but still goes over HTTP.
                embedding = await ollama_client.embed(chunk)
                qdrant_client.upsert_chunk(
                    doc_id=doc_id,
                    chunk_index=index,
                    text=chunk,
                    source=original_filename,
                    embedding=embedding,
                )

                # Entity extraction is the heaviest single operation (~1-5s).
                # This is where most of the ingestion time is spent, and
                # therefore where parallelism gives the biggest speedup.
                graph = await extract_entities_relations(chunk)
                neo4j_client.write_chunk_graph(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=index,
                    source=original_filename,
                    text=chunk,
                    entities=graph["entities"],
                    relations=graph["relations"],
                )

                # Update progress atomically. Chunks finish out of order,
                # so we can't use the loop index — we count completions.
                async with progress_lock:
                    completed_count += 1
                    update_job(job_id, processed_chunks=completed_count)

        # Launch all chunks concurrently (semaphore bounds actual parallelism).
        # asyncio.gather preserves the "wait for all to finish" semantics.
        await asyncio.gather(
            *(_process_chunk(i, chunk) for i, chunk in enumerate(chunks))
        )

        update_job(job_id, status="done")
    except Exception as exc:  # noqa: BLE001 - report to the job, don't crash the worker
        logger.exception("Ingestion failed for %s", original_filename)
        update_job(job_id, status="error", error=str(exc))
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass
