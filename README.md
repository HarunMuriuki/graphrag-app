# GraphRAG

A self-hosted, fully local Graph RAG (Retrieval-Augmented Generation) application. It ingests
your documents, builds a knowledge graph from them alongside vector embeddings, and answers
questions by combining semantic (vector) search with graph traversal -- then shows you the
actual subgraph it used to answer each question.

Everything runs in Docker, on your own machine, with no external API calls: local LLM +
embeddings via [Ollama](https://ollama.com) (Mistral 7B + nomic-embed-text), vector search via
[Qdrant](https://qdrant.tech), and the knowledge graph in [Neo4j](https://neo4j.com).

This is the graph-based evolution of a simpler ("naive") RAG pipeline that used only Ollama +
Qdrant. The main addition here is Neo4j: during ingestion, an LLM extracts entities and
relationships from each chunk into a graph, and at query time the app expands out from the
chunks a vector search finds to pull in connected facts -- which is what lets it answer
questions that span multiple documents or require connecting two facts that never appear in
the same passage.

## Stack

| Layer            | Technology                                            |
|-------------------|--------------------------------------------------------|
| LLM + embeddings  | Ollama (`mistral` for generation & extraction, `nomic-embed-text` for embeddings) |
| Vector store      | Qdrant                                                |
| Knowledge graph   | Neo4j (Community Edition + APOC)                      |
| Backend API       | Python, FastAPI                                       |
| Frontend          | Next.js (App Router) + TypeScript + Tailwind CSS, `react-force-graph-2d` for the graph view |
| Orchestration     | Docker Compose                                        |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit together and why.

## Prerequisites

- Docker and Docker Compose (Docker Desktop on Mac/Windows, or Docker Engine + the `compose`
  plugin on Linux).
- At least ~10GB of free disk space (Ollama models + Neo4j + Qdrant data) and 8GB+ of RAM
  free for the containers -- Mistral 7B is not tiny.
- Internet access **for the first run only**, to pull the Ollama models and Docker images.
  After that, everything runs offline.

## Quickstart

```bash
git clone <this project> graphrag-app   # or just unzip it
cd graphrag-app
cp .env.example .env                    # optional -- defaults work out of the box
docker compose build
docker compose up
```

Then open:

- **http://localhost:3000** -- the app (chat + upload + graph view)
- **http://localhost:8000/docs** -- backend API docs (FastAPI's Swagger UI)
- **http://localhost:7474** -- Neo4j Browser, if you want to poke at the graph directly with
  Cypher (login with the credentials in `.env`, default `neo4j` / `graphrag_password`)

### First run will take a while

On `docker compose up`, a one-shot `model-init` service pulls `mistral` (~4.4GB) and
`nomic-embed-text` (~275MB) into a shared volume before the backend is allowed to start --
you'll see it logging progress. Depending on your connection this can take several minutes to
~20 minutes. Subsequent `docker compose up` runs are fast, since the models persist in the
`ollama_data` volume.

The backend itself also waits for Qdrant and Neo4j to report healthy before initializing, and
retries its own setup calls for a while if they're not, so a slow first boot shouldn't cause
crash loops -- just patience.

## Using it

1. Open http://localhost:3000.
2. Upload one or more documents from the sidebar (PDF, DOCX, TXT, Markdown, and most other
   plain-text formats are supported -- see [Supported file types](#supported-file-types)).
   Watch the job status update as it chunks, embeds, and extracts entities/relationships from
   each chunk. Larger documents take longer, since each chunk requires one embedding call and
   one LLM extraction call.
3. Once ingestion finishes, ask a question in the chat box.
4. As the answer streams in, the right-hand panel switches from the whole-graph overview to
   just the entities and relationships that were actually pulled into context for that
   answer. Each answer's message bubble also has a collapsible "N source passage(s)" section
   showing the raw text chunks that were retrieved.

### Supported file types

Explicit parsers exist for **PDF**, **DOCX**, and plain-text formats (**TXT**, **Markdown**,
CSV, JSON, HTML, code files, etc.). Anything else is still accepted -- the app falls back to a
best-effort text decode, which works for most text-based formats but will (correctly) fail
with a clear error for genuinely binary files (images, audio, compiled binaries, and so on),
since there's no text in those to build a graph or embeddings from.

## Configuration

All configuration is environment variables, set via `.env` (copy `.env.example`) or directly
in `docker-compose.yml`. Key ones:

| Variable                 | Default                      | Notes |
|---------------------------|------------------------------|-------|
| `LLM_MODEL`               | `mistral`                    | Any Ollama-pullable chat model tag. |
| `EMBEDDING_MODEL`         | `nomic-embed-text`           | If you change this, also update `embedding_dim` in `backend/app/core/config.py` to match the new model's output size, and recreate the Qdrant collection (it's created with a fixed vector size). |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `graphrag_password` | Change the password before exposing this beyond your own machine. |
| `NEXT_PUBLIC_API_URL`    | `http://localhost:8000`      | The URL your **browser** uses to reach the backend. Baked into the frontend at build time -- if you change it, run `docker compose build frontend` again. |
| `CORS_ALLOW_ORIGINS`     | `*`                           | Tighten this if you deploy anywhere less trusted than localhost. |

Other tunables (chunk size/overlap, vector top-k, graph expansion hops, max upload size) live
in `backend/app/core/config.py` with inline comments.

## Project structure

```
graphrag-app/
├── docker-compose.yml       # ollama, model-init, qdrant, neo4j, backend, frontend
├── .env.example
├── backend/                 # FastAPI app
│   ├── app/
│   │   ├── core/            # settings, pydantic schemas
│   │   ├── db/              # Ollama / Qdrant / Neo4j clients
│   │   ├── ingestion/       # file loaders, chunker, LLM entity/relation extractor, pipeline, job tracker
│   │   ├── retrieval/       # vector search, graph search, hybrid combiner
│   │   ├── generation/      # prompt building, streaming LLM call
│   │   └── routers/         # /ingest, /chat, /graph, /documents, /health
│   └── Dockerfile
├── frontend/                 # Next.js app
│   └── app/
│       ├── components/      # ChatWindow, MessageBubble, UploadPanel, DocumentList, GraphPanel
│       └── lib/              # API client, shared types
└── docs/
    └── ARCHITECTURE.md
```

## How retrieval actually works (short version)

1. Your question is embedded with `nomic-embed-text` and matched against chunk embeddings in
   Qdrant -> top-k most relevant text passages.
2. The backend looks up which entities were extracted from those specific chunks (via a
   `MENTIONS` relationship in Neo4j), then expands outward through the graph's `RELATED`
   relationships to pull in connected facts -- including ones that live in other chunks or
   other documents entirely.
3. Both the raw passages and the graph facts are assembled into one prompt and sent to
   Mistral, which is instructed to answer using only that context.
4. The subgraph used (seed entities + their expansion) is sent back to the frontend
   alongside the answer, so you can see exactly what the graph contributed.

Full details, including the Neo4j schema and the reasoning behind the hybrid (Qdrant + Neo4j)
design instead of Neo4j alone, are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Troubleshooting

- **`model-init` seems stuck / backend never starts.** Check `docker compose logs model-init
  -f`. It's almost always still downloading -- Mistral 7B is several GB. If it's genuinely
  stuck (no log output for a long time), restart it: `docker compose restart model-init`.
- **Neo4j container keeps restarting / healthcheck failing.** Neo4j needs a bit of memory;
  make sure Docker has at least 4GB allocated to it, and check `docker compose logs neo4j`.
- **Ports already in use.** 3000, 8000, 6333, 7474, 7687, and 11434 all need to be free on
  your host. Change the left-hand side of the relevant `ports:` mapping in
  `docker-compose.yml` if something else is using one of them (and update
  `NEXT_PUBLIC_API_URL` accordingly if you move the backend's port).
- **Ingestion job shows `error`.** Check `docker compose logs backend`; the error message on
  the job itself (shown under the upload item in the sidebar) usually explains it directly
  (unsupported/binary file, or an Ollama call failing).
- **Want a clean slate?** `docker compose down -v` removes all containers *and* the
  Ollama/Qdrant/Neo4j data volumes -- you'll re-download models and lose all ingested data.

## Known limitations (by design, for a local/single-user app)

- Ingestion jobs are tracked in-memory in the backend process -- restarting the backend loses
  in-flight job status (already-ingested data in Qdrant/Neo4j is unaffected).
- Entity/relationship extraction quality depends on Mistral 7B's ability to follow the
  extraction prompt -- it's good, not perfect. Bigger models (swap `LLM_MODEL`) will extract
  a richer, more accurate graph at the cost of slower ingestion and generation.
- No authentication anywhere in this stack. It's built to run on your own machine; don't
  expose these ports to the open internet as-is.
