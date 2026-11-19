# BM25 + Reciprocal Rank Fusion (RRF) Hybrid Search — Implementation Plan

Branch: `modelBM25`
Base: `modelTest` (d6e86f6)

## Why we're doing this (the problem)

Your current retrieval pipeline is **dense-vector-only**:

```
query -> embed with nomic-embed-text -> Qdrant vector search -> top_k chunks
```

This is why you rated answer quality 35/100. Dense embeddings are great at *semantic*
similarity (paraphrase, concept matching) but **poor at exact-match recall** — the
exact field names, error codes, account types, and numbers that Finacle banking scripts
are full of. A 2026 benchmark on *financial* documents confirmed this directly:

> "BM25 outperformed state-of-the-art dense retrieval on financial documents...
> financial documents are full of identifiers, numbers, and exact terminology
> that dense embeddings smooth over." (arxiv 2604.01733)

**The fix:** run BOTH a sparse keyword search (BM25) AND the dense vector search in
parallel, then fuse the two ranked lists with **Reciprocal Rank Fusion (RRF)**. A chunk
that appears in *both* lists scores much higher than one that only appears in one.

Research shows hybrid + RRF reliably beats either alone (~7-17% quality lift), and it
doubles the effective retrieval signal for zero extra LLM cost.

## Your exact stack — and why Qdrant makes this easy

- `qdrant-client==1.11.2` — this version supports the **Query API with hybrid search**
  (dense + sparse + RRF in a single call: `client.query_points` with `QueryRequest`,
  `prefetch` + `Fusion.RRF`).
- Qdrant's sparse BM25 uses its **built-in sparse vector support** — no extra Elasticsearch
  needed. We just add a **sparse vector field** to the collection and store a BM25-style
  sparse index for each chunk.
- Ollama handles the dense embeddings (unchanged).

## The critical decision: sparse vector generation

Qdrant's native sparse indexing needs a **sparse vector** (a map of token-id → weight) per
chunk. There are two ways to produce it at ingestion time:

1. **Deterministic term-frequency hashing (BM25 weights) computed in-app** — we tokenize
   the chunk text ourselves, hash tokens into Qdrant's sparse-vector space (it uses a
   MurmurHash scheme internally when given raw text, but for explicit control we compute
   weights with a classic BM25 term-frequency formula). This is **offline, free, CPU-only,
   and fast**. This is what we'll do.

2. **Ollama/espell LLM sparse embedding** — not applicable/slower. Skip.

We will compute each chunk's sparse vector at ingestion time (alongside the dense
embedding) and store it as a second vector field named `"sparse"`.

## Files we will change

### 1. `backend/app/core/config.py`
Add settings for the new retrieval path (with sensible defaults):
- `vector_top_k` already exists (13) — keep as the dense candidate count.
- Add `bm25_weight` and `rrf_k` fusion knobs so you can tune without code edits.
- Possibly add a `use_hybrid: bool = True` flag to toggle hybrid on/off for A/B testing.

### 2. `backend/app/db/qdrant_client.py`  ← **heaviest change**
- `ensure_collection()`: create the collection with **two vector fields**:
  - `"dense"`: size=embedding_dim, COSINE (existing behaviour)
  - `"sparse"`: sparse vector config for BM25
- `upsert_chunk()`: also accept and store the sparse vector in the point.
- New `sparse_search(query_text, top_k)`: run a pure BM25 keyword search using the
  sparse field.
- New `hybrid_search(query_embedding, query_text, top_k)`:
  - Build a `QueryRequest` with a `prefetch` list containing BOTH the dense prefetch
    and the `"sparse"` prefetch, fused with `Fusion.RRF`.
  - Return the fused, deduplicated chunk list.

### 3. `backend/app/retrieval/vector_search.py`
- Add `bm25_sparse_terms(text)` helper: tokenize + hash tokens into Qdrant sparse space
  and compute BM25 term-frequency weights. (Or place this in `qdrant_client` / a small
  `sparse.py` module — I'll put it in a dedicated `backend/app/ingestion/sparse.py` so it
  can be shared by both ingestion (ingest) and retrieval (query).)
- `search_chunks()`: switch from dense-only to `hybrid_search` (dense + RRF).
- Keep returning the same dict shape `{score, chunk_id, doc_id, chunk_index, text, source}`
  so `hybrid.py` and `chat.py` don't need to change.

### 4. `backend/app/ingestion/pipeline.py`
- Compute the sparse vector for each chunk during the embedding phase (right where we
  already compute the dense embedding) and pass it to `upsert_chunk`.
- No new LLM calls — this is pure token math on CPU.

### 5. New file `backend/app/ingestion/sparse.py` (or `vector_search.py`)
- Shared tokenizer + BM25 sparse vector builder used at BOTH ingestion (build sparse
  index) and query time (build sparse query vector). Single source of truth so the
  token→hash mapping is consistent between index and query (critical for matching).

### 6. `backend/app/generation/prompt.py` (minor, optional)
- The prompt already lists passages. With hybrid retrieval the retrieved set is better,
  so no structural change needed. Only update if we want to surface a confidence/score.

## What stays the same (no changes needed)

- `retrieval/hybrid.py` — calls `search_chunks()` which keeps its signature.
- `routers/chat.py` — consumes the same chunk dict shape.
- `generation/llm.py`, `prompt.py` — unchanged logic.
- Frontend — unchanged; the graph panel + sources list work off the same payload.

## Re-ingestion requirement (IMPORTANT)

Switching the collection to dual vectors (dense + sparse) changes the collection schema.
**Existing Qdrant points have no sparse vector**, so you **must delete all existing
documents and re-ingest** for hybrid search to return results. The collection can be
recreated on startup by `ensure_collection()` (drop + recreate, or delete the collection
volume). This is expected — the whole point is a fresh, better-indexed corpus.

## Tuning knobs (so you can experiment)

- `rrf_k` — RRF constant (default 60, standard value from the literature). Higher = more
  emphasis on rank position differences.
- `bm25_sparse_limit` — how many top sparse hits to return before fusion (candidate pool).
- `hybrid_top_k` — final fused result count fed to the LLM.

## Suggested verification

1. Delete existing docs; re-ingest a small Finacle file (sentence strategy).
2. Check Qdrant collection has both `dense` and `sparse` vectors populated.
3. Ask a query with an **exact Finacle term/field name** (e.g. a specific account-type
   code) and confirm the BM25 path returns chunks the dense path missed.
4. A/B toggle `use_hybrid` on/off and compare answer quality + which sources the LLM sees.

## Out of scope (deferred)

- Community detection / Leiden / LazyGraphRAG-style summarization (global queries).
- Cross-encoder rerankers (adds complexity; revisit after hybrid proves itself).
- Switching to NLP co-occurrence extraction (FastGraphRAG) — separate optimization.
