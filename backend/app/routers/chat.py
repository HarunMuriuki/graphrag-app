"""
The main question-answering endpoint.

POST /chat/stream streams Server-Sent Events back to the frontend:

  event: context   -- sent once, right after retrieval finishes. data is a
                       JSON object {sources: [...], graph: {nodes, edges}}.
                       The frontend uses this to populate the graph panel and
                       the "sources used" list immediately, before the answer
                       has finished generating.
  event: token      -- sent repeatedly as the LLM generates the answer.
                       data is a JSON-encoded string containing one token/
                       text fragment.
  event: done       -- sent once, when generation is complete.
  event: error      -- sent if something goes wrong mid-stream.

We use a hand-rolled SSE format (rather than a library) over a plain
fetch()+ReadableStream on the frontend, since EventSource does not support
POST bodies and we need to send the question in the request body.
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.schemas import ChatRequest
from app.generation.llm import stream_answer
from app.generation.prompt import build_prompt
from app.retrieval.hybrid import retrieve

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            top_k = req.top_k or settings.vector_top_k
            result = await retrieve(req.message, top_k=top_k)
            chunks = result["chunks"]
            graph = result["graph"]

            context_payload = {
                "sources": [
                    {
                        "chunk_id": c["chunk_id"],
                        "source": c["source"],
                        "text": c["text"],
                        "score": c["score"],
                    }
                    for c in chunks
                ],
                "graph": {"nodes": graph["nodes"], "edges": graph["edges"]},
            }
            yield _sse("context", context_payload)

            prompt = build_prompt(req.message, chunks, graph["edges"])
            async for token in stream_answer(prompt):
                yield _sse("token", json.dumps(token))

            yield _sse("done", {})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", json.dumps(str(exc)))

    return StreamingResponse(event_generator(), media_type="text/event-stream")
