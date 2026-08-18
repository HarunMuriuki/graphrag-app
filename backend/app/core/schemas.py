from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float


class GraphNode(BaseModel):
    id: str
    label: str
    type: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    description: str | None = None


class GraphPayload(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ChatContext(BaseModel):
    sources: list[SourceChunk]
    graph: GraphPayload


class IngestJobStatus(BaseModel):
    id: str
    filename: str
    status: str
    total_chunks: int
    processed_chunks: int
    error: str | None = None


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    chunks: int
