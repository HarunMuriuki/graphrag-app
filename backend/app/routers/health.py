from fastapi import APIRouter

from app.db import neo4j_client, qdrant_client
from app.db.ollama_client import ollama_client

router = APIRouter()


@router.get("/health")
async def health():
    ollama_ok = await ollama_client.ping()

    try:
        qdrant_client.get_qdrant().get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    neo4j_ok = neo4j_client.ping()

    overall = ollama_ok and qdrant_ok and neo4j_ok
    return {
        "status": "ok" if overall else "degraded",
        "dependencies": {
            "ollama": ollama_ok,
            "qdrant": qdrant_ok,
            "neo4j": neo4j_ok,
        },
    }
