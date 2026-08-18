from fastapi import APIRouter

from app.db import neo4j_client

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/overview")
async def graph_overview(limit: int = 150):
    """A sample of the whole knowledge graph, used by the frontend to render
    something in the graph panel before any question has been asked."""
    return neo4j_client.get_graph_sample(limit=limit)
