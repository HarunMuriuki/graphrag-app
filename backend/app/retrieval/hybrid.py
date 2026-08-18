"""
Combines the vector search step (which chunks of text are semantically
closest to the question) with the graph search step (which entities/facts
are connected to those chunks in the knowledge graph) into a single retrieval
result the generation step can turn into a prompt, and the frontend can turn
into a graph visualization.
"""
from app.retrieval.graph_search import graph_context_for_chunks
from app.retrieval.vector_search import search_chunks


async def retrieve(query: str, top_k: int) -> dict:
    chunks = await search_chunks(query, top_k=top_k)
    chunk_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
    graph = graph_context_for_chunks(chunk_ids)
    return {"chunks": chunks, "graph": graph}
