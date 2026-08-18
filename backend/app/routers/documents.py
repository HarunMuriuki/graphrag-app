from fastapi import APIRouter, HTTPException

from app.db import neo4j_client, qdrant_client

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents():
    return qdrant_client.list_sources()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all of its data: Qdrant embeddings, Neo4j chunks,
    MENTIONS edges, and orphaned Entity nodes. Returns the number of chunks
    removed from each store."""
    # Verify the document exists before attempting deletion
    docs = qdrant_client.list_sources()
    if not any(d["doc_id"] == doc_id for d in docs):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    qdrant_deleted = qdrant_client.delete_points_by_doc_id(doc_id)
    neo4j_deleted = neo4j_client.delete_document_graph(doc_id)
    return {
        "doc_id": doc_id,
        "qdrant_chunks_deleted": qdrant_deleted,
        "neo4j_chunks_deleted": neo4j_deleted,
    }
