export type Role = "user" | "assistant";

export interface SourceChunk {
  chunk_id: string;
  source: string;
  text: string;
  score: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type?: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  description?: string;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources?: SourceChunk[];
  isStreaming?: boolean;
}

export interface IngestJobStatus {
  id: string;
  filename: string;
  status: "pending" | "processing" | "done" | "error";
  total_chunks: number;
  processed_chunks: number;
  error?: string | null;
}

export interface DocumentSummary {
  doc_id: string;
  source: string;
  chunks: number;
}
