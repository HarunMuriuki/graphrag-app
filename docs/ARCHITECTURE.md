# Architecture

## Why hybrid (Qdrant + Neo4j) instead of Neo4j alone

Neo4j can store vector embeddings directly on nodes and query them with a native vector
index, which would let one database do both jobs. This project instead keeps a dedicated
vector store (Qdrant) alongside the graph, because:

- Vector similarity search and graph traversal are different workloads, and Qdrant's HNSW
  index is more mature and configurable for the former than Neo4j's vector index.
- It keeps the two concerns cleanly separated in the codebase: `app/retrieval/vector_search.py`
  only knows about chunk similarity, `app/retrieval/graph_search.py` only knows about entity
  relationships. `app/retrieval/hybrid.py` is the only place that combines them.
- It mirrors how most production "Graph RAG" systems are actually built in practice: a vector
  store for recall over raw text, a graph for the structured relationships that vector search
  alone can't surface (e.g. two facts about the same entity that appear in unrelated
  documents, or transitive relationships like "A supplies B, B is part of C, so A indirectly
  supplies C").

## Data model

**Qdrant** (`graphrag_chunks` collection): one point per chunk.

```
id:      deterministic UUID = uuid5(doc_id + chunk_index)
vector:  embedding from nomic-embed-text (768-dim)
payload: { doc_id, chunk_id, chunk_index, text, source }
```

**Neo4j**:

```
(:Chunk {id, doc_id, source, chunk_index, text})
(:Entity {name, type})

(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:RELATED {type, description}]->(:Entity)
```

`Chunk.id` is the *same* UUID as the Qdrant point id -- that shared key is what lets retrieval
hop from "vector search matched this chunk" to "here's what the graph knows about the
entities in that chunk." `Entity.name` is the natural key (a uniqueness constraint is created
on it at startup); relationships all use the generic Neo4j relationship type `RELATED`, with
the actual semantic label (e.g. `works_at`, `located_in`) stored as a property. This keeps the
graph schema stable while letting the LLM's extraction vocabulary be arbitrary -- you don't
need to predefine every possible relationship type up front.

## Ingestion pipeline (`app/ingestion/pipeline.py`)

```
upload -> extract_text (loaders.py: PDF/DOCX/plain-text)
       -> chunk_text (chunker.py: ~1000 chars, 150 char overlap)
       -> for each chunk, in sequence:
            embed(chunk)              -> upsert into Qdrant
            extract_entities_relations(chunk)  -> write into Neo4j
```

Entity/relationship extraction (`app/ingestion/extractor.py`) prompts Mistral, with Ollama's
`format: json` constrained decoding, to return `{"entities": [...], "relations": [...]}` for
each chunk. This runs once per chunk, so ingestion time scales with document length and is
dominated by LLM extraction calls, not embedding (embeddings are much cheaper). If a chunk's
extraction fails or returns malformed JSON, that chunk simply contributes no graph data (its
text is still embedded and searchable) -- a single bad extraction never aborts the whole job.

Ingestion runs as a FastAPI `BackgroundTask` with progress tracked in an in-memory job store
(`app/ingestion/jobs.py`), polled by the frontend via `GET /ingest/status/{job_id}`.

## Retrieval pipeline (`app/retrieval/hybrid.py`)

```
question
  -> embed(question) -> Qdrant search (top-k chunks)
  -> Neo4j: MATCH (Chunk)-[:MENTIONS]->(Entity) for those chunk ids  (seed entities)
  -> Neo4j: BFS expansion 1..N hops out from seed entities over :RELATED  (app/db/neo4j_client.py: expand_subgraph)
  -> merge: chunk texts + graph facts (rendered as "A --[rel]--> B (description)")
  -> prompt (app/generation/prompt.py) -> Mistral, streamed token-by-token
```

The graph expansion is implemented as a small Python BFS loop issuing one Cypher query per
hop, rather than a single variable-length-path Cypher query. This was a deliberate
simplification: Neo4j doesn't cleanly support parameterizing the hop-count bound in a
variable-length pattern (`*1..N`) across driver versions, and a hop-at-a-time loop is easy to
cap (`graph_max_nodes` in config) so the returned subgraph stays small enough to actually
render in the UI.

## Generation & streaming (`app/generation/`, `app/routers/chat.py`)

`POST /chat/stream` returns a hand-rolled Server-Sent Events stream (not the browser's native
`EventSource`, since that only supports `GET` and we need to POST the question body):

```
event: context   -- once, right after retrieval: { sources: [...], graph: { nodes, edges } }
event: token      -- repeated, one per generated token/fragment
event: done       -- once, generation complete
event: error      -- on failure
```

Sending `context` before generation starts is what lets the frontend populate the graph panel
and "sources used" list immediately, rather than waiting for the full answer -- useful since
Mistral 7B generation can take a while for longer answers.

## Frontend (`frontend/app/`)

Three-column layout: upload/document sidebar, chat column, graph panel. The graph panel
(`components/GraphPanel.tsx`) uses `react-force-graph-2d`, which touches `window`/canvas at
import time and is therefore loaded via `next/dynamic(..., { ssr: false })`. It shows the
whole-graph overview (`GET /graph/overview`) until the first question is answered, then
switches to the subgraph returned in that answer's `context` event.

## Extending this

- **Bigger/different LLM:** change `LLM_MODEL` / `EMBEDDING_MODEL` in `.env`. If you change
  the embedding model to one with a different output dimension, also update
  `embedding_dim` in `backend/app/core/config.py` and recreate the Qdrant collection (it's
  created once at startup with a fixed vector size, via `ensure_collection()`).
- **Multi-hop reasoning depth:** `graph_expand_hops` in config controls how many hops the BFS
  expansion walks out from seed entities. Higher values surface more distant connections at
  the cost of noisier context and a slower prompt.
- **Swap the vector store:** `app/db/qdrant_client.py` and `app/retrieval/vector_search.py`
  are the only files that know Qdrant exists; everything else talks in plain dicts.
- **Persistent job tracking:** `app/ingestion/jobs.py` is a small, swappable interface
  (`create_job` / `get_job` / `update_job` / `list_jobs`) currently backed by an in-memory
  dict -- swap the implementation for Redis/Postgres if you need job history to survive a
  backend restart.
