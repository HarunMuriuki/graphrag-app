from fastapi import APIRouter

from app.db import qdrant_client

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents():
    return qdrant_client.list_sources()
