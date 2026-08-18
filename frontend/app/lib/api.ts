import type { DocumentSummary, GraphPayload, IngestJobStatus, SourceChunk } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export async function uploadDocument(file: File): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/ingest/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<IngestJobStatus> {
  const res = await fetch(`${API_BASE}/ingest/status/${jobId}`);
  if (!res.ok) throw new Error(`Failed to fetch job status: ${res.status}`);
  return res.json();
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error(`Failed to list documents: ${res.status}`);
  return res.json();
}

export async function deleteDocument(docId: string): Promise<{
  doc_id: string;
  qdrant_chunks_deleted: number;
  neo4j_chunks_deleted: number;
}> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete document: ${res.status}`);
  return res.json();
}

export async function getGraphOverview(limit = 150): Promise<GraphPayload> {
  const res = await fetch(`${API_BASE}/graph/overview?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch graph overview: ${res.status}`);
  return res.json();
}

export interface ChatStreamHandlers {
  onContext?: (ctx: { sources: SourceChunk[]; graph: GraphPayload }) => void;
  onToken?: (token: string) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

/**
 * Consumes the backend's hand-rolled SSE stream from POST /chat/stream.
 * We can't use the native EventSource API because it only supports GET
 * requests, and we need to POST the question in the request body.
 */
export async function streamChat(message: string, handlers: ChatStreamHandlers): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok || !res.body) {
    handlers.onError?.(`Chat request failed: ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleSseEvent(rawEvent, handlers);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function handleSseEvent(rawEvent: string, handlers: ChatStreamHandlers): void {
  let eventType = "message";
  let data = "";
  for (const line of rawEvent.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;

  try {
    switch (eventType) {
      case "context":
        handlers.onContext?.(JSON.parse(data));
        break;
      case "token":
        handlers.onToken?.(JSON.parse(data));
        break;
      case "done":
        handlers.onDone?.();
        break;
      case "error":
        handlers.onError?.(JSON.parse(data));
        break;
    }
  } catch {
    // ignore malformed event, stream continues
  }
}
