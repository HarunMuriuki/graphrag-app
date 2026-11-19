"""Retrieval step: embed the query, then search Qdrant.

HYBRID SEARCH
=============
We run BOTH a dense (semantic) search and a sparse (BM25 keyword) search, fused
with Reciprocal Rank Fusion. This is important for Finacle documents, which are
full of exact field names, codes, and numbers that dense embeddings tend to
smooth over. Hybrid retrieval lets exact keyword matches reinforce semantic
matches.

The output dict shape is unchanged ({score, chunk_id, doc_id, chunk_index, text,
source}) so downstream code (hybrid.py, chat.py) does not need to change.
"""
from app.core.config import settings
from app.db import qdrant_client
from app.db.ollama_client import ollama_client
from app.ingestion.sparse import build_sparse_query


async def search_chunks(query: str, top_k: int = settings.vector_top_k) -> list[dict]:
    # Dense embedding of the question (nomic-embed-text).
    embedding = await ollama_client.embed(query)

    # Sparse (BM25) representation of the question -- pure token math, no LLM.
    # Using the same tokenizer/hasher as ingestion keeps index and query aligned.
    sparse = build_sparse_query(query)

    # Route to the configured search mode so we can A/B test easily.
    if settings.use_hybrid:
        return qdrant_client.hybrid_search(
            embedding,
            sparse,
            top_k=top_k,
            dense_limit=settings.vector_top_k,
            sparse_limit=settings.bm25_sparse_limit,
        )
    # Fall back to dense-only (previous behaviour) for comparison.
    return qdrant_client.search(embedding, top_k=top_k)
