"""Semantic (vector) retrieval step: embed the query, search Qdrant."""
from app.core.config import settings
from app.db import qdrant_client
from app.db.ollama_client import ollama_client


async def search_chunks(query: str, top_k: int = settings.vector_top_k) -> list[dict]:
    embedding = await ollama_client.embed(query)
    return qdrant_client.search(embedding, top_k=top_k)
