import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db import neo4j_client, qdrant_client
from app.routers import chat, documents, graph, health, ingest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag.startup")

app = FastAPI(title="GraphRAG API", version="1.0.0")

origins = [o.strip() for o in settings.cors_allow_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(graph.router)
app.include_router(documents.router)


async def _wait_and_init(name: str, fn, retries: int = 20, delay: float = 3.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            fn()
            logger.info("%s ready.", name)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s not ready yet (attempt %d/%d): %s", name, attempt, retries, exc)
            await asyncio.sleep(delay)
    logger.error("%s never became ready after %d attempts; continuing anyway.", name, retries)


@app.on_event("startup")
async def on_startup():
    logger.info("Initializing Qdrant collection and Neo4j constraints...")
    await _wait_and_init("Qdrant", qdrant_client.ensure_collection)
    await _wait_and_init("Neo4j", neo4j_client.init_constraints)


@app.on_event("shutdown")
async def on_shutdown():
    neo4j_client.close_driver()


@app.get("/")
async def root():
    return {
        "name": "GraphRAG API",
        "docs": "/docs",
        "endpoints": ["/health", "/ingest/upload", "/chat/stream", "/graph/overview", "/documents"],
    }
