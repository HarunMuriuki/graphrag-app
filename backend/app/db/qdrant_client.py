"""
Wrapper around the Qdrant client: collection bootstrap, upsert of chunk
embeddings, and hybrid retrieval (dense vector + BM25 sparse + RRF).

HYBRID SEARCH
=============
We store TWO vector fields per chunk in the Qdrant collection:
  * "dense"  -- the nomic-embed-text embedding (semantic similarity).
  * "sparse" -- a BM25 term-frequency sparse vector (exact keyword matching
                on field names, codes, numbers).
At query time we issue a `QueryRequest` with two prefetches (dense + sparse)
fused by Reciprocal Rank Fusion (`Fusion.RRF`). This lets exact-match and
semantic matches reinforce each other -- the key fix for Finacle banking
scripts full of precise identifiers that dense embeddings smooth over.
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
    """Create the collection with BOTH dense and sparse vector fields if missing.

    SCHEMA MIGRATION / RE-INGESTION
    ================================
    Hybrid search requires the "sparse" vector field. If an existing collection
    was created before this change (single dense vector only), it cannot serve
    hybrid results. We detect that and DROP + recreate the collection so the
    sparse field exists -- which means any previously-ingested documents must be
    re-uploaded (their points are discarded with the old collection).
    """
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection in existing:
        info = client.get_collection(settings.qdrant_collection)
        has_sparse = "sparse" in (info.config.params.sparse_vectors or {})
        if not has_sparse:
            # Old single-vector schema: drop it so we can rebuild with sparse.
            import logging

            logging.getLogger("graphrag.startup").warning(
                "Collection '%s' lacks the sparse vector field required for hybrid "
                "search. Recreating it -- please re-ingest your documents.",
                settings.qdrant_collection,
            )
            client.delete_collection(settings.qdrant_collection)

    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": qm.VectorParams(
                    size=settings.embedding_dim, distance=qm.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams()),
            },
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
    sparse: dict[int, float] | None = None,
) -> str:
    """Store a chunk's dense + sparse vectors.

    Parameters
    ----------
    sparse : dict[int, float] | None
        The BM25 sparse vector ({token_id: weight}) computed by
        app.ingestion.sparse for this chunk's text. If None (e.g. legacy call),
        only the dense vector is stored -- hybrid search will skip such points.
    """
    client = get_qdrant()
    point_id = chunk_point_id(doc_id, chunk_index)

    # Build the per-named-vector dict Qdrant expects for multi-vector points.
    vectors: dict[str, Any] = {"dense": embedding}
    if sparse:
        # A SparseVector uses parallel `indices` / `values` arrays.
        indices = sorted(sparse.keys())
        vectors["sparse"] = qm.SparseVector(
            indices=indices, values=[sparse[i] for i in indices]
        )

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=vectors,
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
    """Pure dense (semantic) vector search -- kept for reference / A-B testing."""
    client = get_qdrant()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=("dense", embedding),
        limit=top_k,
        with_payload=True,
    )
    return [_row(r) for r in results]


def _row(r) -> dict[str, Any]:
    """Normalize a Qdrant hit into our stable chunk dict shape."""
    payload = r.payload or {}
    return {
        "score": r.score,
        "chunk_id": payload.get("chunk_id"),
        "doc_id": payload.get("doc_id"),
        "chunk_index": payload.get("chunk_index"),
        "text": payload.get("text"),
        "source": payload.get("source"),
    }


def sparse_search(sparse_query: dict[int, float], top_k: int) -> list[dict[str, Any]]:
    """Pure BM25 (sparse) keyword search over the "sparse" vector field."""
    client = get_qdrant()
    queried = qm.QueryRequest(
        prefetch=[
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=sorted(sparse_query.keys()),
                    values=[sparse_query[i] for i in sorted(sparse_query.keys())],
                ),
                using="sparse",
                limit=top_k,
            )
        ],
        query=qm.Fusion.RRF,
        limit=top_k,
        with_payload=True,
    )
    results = client.query_points(
        collection_name=settings.qdrant_collection, query=queried
    ).points
    return [_row(r) for r in results]


def hybrid_search(
    embedding: list[float],
    sparse_query: dict[int, float],
    top_k: int | None = None,
    dense_limit: int | None = None,
    sparse_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Hybrid dense + BM25 search fused with Reciprocal Rank Fusion (RRF).

    Runs two candidate retrievals in a single QueryRequest:
      1. "dense"  prefetch using the named dense vector.
      2. "sparse" prefetch using the named BM25 sparse vector.
    Then fuses them with Qdrant's native Fusion.RRF. A point that appears in
    both lists gets a much higher fused score than one that appears in only a
    single list -- which is exactly the reinforcement we want between semantic
    and exact matches.

    Arguments
    ---------
    embedding      : dense query embedding from Ollama.
    sparse_query   : {token_id: weight} from app.ingestion.sparse.build_sparse_query.
    top_k          : final fused count (default settings.hybrid_top_k).
    dense_limit    : candidate count from the dense path (default settings.vector_top_k).
    sparse_limit   : candidate count from the sparse path (default settings.bm25_sparse_limit).
    """
    top_k = top_k or settings.hybrid_top_k
    dense_limit = dense_limit or settings.vector_top_k
    sparse_limit = sparse_limit or settings.bm25_sparse_limit

    client = get_qdrant()

    prefetches = [
        qm.Prefetch(query=embedding, using="dense", limit=dense_limit),
    ]
    if sparse_query:
        # Only add the BM25 path if we actually have query terms.
        prefetches.append(
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=sorted(sparse_query.keys()),
                    values=[sparse_query[i] for i in sorted(sparse_query.keys())],
                ),
                using="sparse",
                limit=sparse_limit,
            )
        )

    queried = qm.QueryRequest(
        prefetch=prefetches,
        query=qm.Fusion.RRF,
        limit=top_k,
        with_payload=True,
    )

    results = client.query_points(
        collection_name=settings.qdrant_collection, query=queried
    ).points
    return [_row(r) for r in results]


def delete_points_by_doc_id(doc_id: str) -> int:
    """Delete all Qdrant points belonging to a document. Returns the number
    of points deleted. Uses a filter scroll + batch delete since Qdrant
    doesn't support a direct 'delete by payload filter' in one call."""
    client = get_qdrant()
    # Collect all point IDs for this doc_id
    point_ids: list[str] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=next_offset,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            ),
            with_payload=False,
        )
        point_ids.extend(p.id for p in points)
        if next_offset is None:
            break

    if point_ids:
        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=qm.PointIdsList(points=point_ids),
        )
    return len(point_ids)


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
