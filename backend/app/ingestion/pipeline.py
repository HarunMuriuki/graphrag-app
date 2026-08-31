"""
End-to-end ingestion pipeline, run as a FastAPI BackgroundTask:

    load file -> extract text -> chunk ->
        PHASE 1: embed ALL chunks -> upsert into Qdrant  (embedding model loaded)
        PHASE 2: extract entities/relations for ALL chunks -> write into Neo4j  (LLM loaded)

Every chunk gets a stable UUID (derived from doc_id + chunk_index) that is
shared between Qdrant (point id) and Neo4j (Chunk.id), which is what lets
retrieval hop from "these chunks matched the vector search" to "these
entities/relationships are attached to those chunks" in the graph.

TWO-PHASE PROCESSING:
    Ollama swaps models between embedding (nomic-embed-text) and generation
    (phi4-mini). Each swap takes ~30-60s of model loading time. By batching
    all embeddings first, then all entity extractions, we force only ONE
    model swap instead of one per chunk. For a 143-chunk document, that
    saves ~143 * 30s = ~70 minutes of wasted model loading time.

    Within each phase, chunks are processed concurrently (bounded by
    ingestion_concurrency semaphore). Progress is tracked as a single
    counter that advances through both phases.
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
        # Pass Ollama embed function for semantic chunking strategy;
        # ignored by recursive/sentence strategies
        chunks = await chunk_text(text, embed_fn=ollama_client.embed)
        update_job(job_id, total_chunks=len(chunks))

        if not chunks:
            update_job(job_id, status="error", error="No extractable text found in file.")
            return

        semaphore = asyncio.Semaphore(settings.ingestion_concurrency)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # ------------------------------------------------------------------
        # PHASE 1: Embedding — all chunks in parallel while the embedding
        # model (nomic-embed-text) is loaded in Ollama. Each chunk gets
        # embedded and upserted into Qdrant. We store the chunk_id for
        # Phase 2.
        # ------------------------------------------------------------------
        logger.info("Phase 1/2: embedding %d chunks", len(chunks))
        update_job(job_id, phase="embed")

        chunk_ids: list[str] = [qdrant_client.chunk_point_id(doc_id, i) for i in range(len(chunks))]

        async def _embed_one(index: int, chunk: str) -> None:
            nonlocal completed_count
            async with semaphore:
                embedding = await ollama_client.embed(chunk)
                qdrant_client.upsert_chunk(
                    doc_id=doc_id,
                    chunk_index=index,
                    text=chunk,
                    source=original_filename,
                    embedding=embedding,
                )
                async with progress_lock:
                    completed_count += 1
                    update_job(job_id, processed_chunks=completed_count)

        await asyncio.gather(*(_embed_one(i, c) for i, c in enumerate(chunks)))
        logger.info("Phase 1 complete: all %d chunks embedded", len(chunks))

        # ------------------------------------------------------------------
        # PHASE 2: Entity extraction — all chunks in parallel while the LLM
        # (phi4-mini) is loaded in Ollama. Ollama will have swapped from
        # the embedding model to the LLM once here, instead of 143 times.
        # Uses a separate (lower) concurrency because LLM generation is
        # CPU-bound — running too many in parallel causes contention and
        # timeouts.
        # ------------------------------------------------------------------
        logger.info("Phase 2/2: extracting entities from %d chunks (concurrency=%d)",
                     len(chunks), settings.extraction_concurrency)
        update_job(job_id, phase="extract")
        extract_semaphore = asyncio.Semaphore(settings.extraction_concurrency)

        async def _extract_one(index: int, chunk: str) -> None:
            nonlocal completed_count
            async with extract_semaphore:
                graph = await extract_entities_relations(chunk)
                neo4j_client.write_chunk_graph(
                    doc_id=doc_id,
                    chunk_id=chunk_ids[index],
                    chunk_index=index,
                    source=original_filename,
                    text=chunk,
                    entities=graph["entities"],
                    relations=graph["relations"],
                )
                async with progress_lock:
                    completed_count += 1
                    update_job(job_id, processed_chunks=completed_count)

        await asyncio.gather(*(_extract_one(i, c) for i, c in enumerate(chunks)))
        logger.info("Phase 2 complete: all %d chunks extracted", len(chunks))

        update_job(job_id, status="done")
    except Exception as exc:  # noqa: BLE001 - report to the job, don't crash the worker
        logger.exception("Ingestion failed for %s", original_filename)
        update_job(job_id, status="error", error=str(exc))
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            pass
