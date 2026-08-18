"""
Wrapper around the Qdrant client: collection bootstrap, upsert of chunk
embeddings, and vector similarity search.
"""
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings

_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection() -> None:
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qm.VectorParams(
                size=settings.embedding_dim, distance=qm.Distance.COSINE
            ),
        )


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic UUID for a (doc_id, chunk_index) pair, so re-ingesting
    the same document overwrites rather than duplicates points."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_index}"))


def upsert_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    source: str,
    embedding: list[float],
) -> str:
    client = get_qdrant()
    point_id = chunk_point_id(doc_id, chunk_index)
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "doc_id": doc_id,
                    "chunk_id": point_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "source": source,
                },
            )
        ],
    )
    return point_id


def search(embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    client = get_qdrant()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
    )
    out = []
    for r in results:
        payload = r.payload or {}
        out.append(
            {
                "score": r.score,
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "source": payload.get("source"),
            }
        )
    return out


def list_sources() -> list[dict[str, Any]]:
    """Scroll the whole collection and aggregate distinct source documents.
    Fine for a demo-scale local app; for large corpora you'd track this in
    a proper metadata store instead."""
    client = get_qdrant()
    sources: dict[str, dict[str, Any]] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=next_offset,
            with_payload=True,
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            entry = sources.setdefault(
                doc_id, {"doc_id": doc_id, "source": payload.get("source"), "chunks": 0}
            )
            entry["chunks"] += 1
        if next_offset is None:
            break
    return list(sources.values())
