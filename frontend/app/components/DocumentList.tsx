"use client";

import { useEffect, useState } from "react";
import { deleteDocument, listDocuments } from "../lib/api";
import type { DocumentSummary } from "../lib/types";

interface Props {
  refreshKey: number;
  onDeleted?: () => void;
}

export default function DocumentList({ refreshKey, onDeleted }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch(() => setDocs([]));
  }, [refreshKey]);

  async function handleDelete(docId: string, source: string) {
    if (!confirm(`Delete "${source}" and all its embeddings?`)) return;
    setDeleting(docId);
    try {
      await deleteDocument(docId);
      setDocs((prev) => prev.filter((d) => d.doc_id !== docId));
      onDeleted?.();
    } catch (err) {
      console.error("Delete failed:", err);
      alert("Failed to delete document.");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Ingested documents ({docs.length})
      </p>
      {docs.length === 0 ? (
        <p className="text-xs text-slate-500">No documents yet.</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {docs.map((d) => (
            <div
              key={d.doc_id}
              className="flex items-center justify-between gap-2 rounded-md bg-slate-800 px-3 py-1.5 text-xs"
            >
              <span className="truncate text-slate-300">{d.source}</span>
              <div className="flex shrink-0 items-center gap-2">
                <span className="text-slate-500">{d.chunks} chunks</span>
                <button
                  onClick={() => handleDelete(d.doc_id, d.source)}
                  disabled={deleting === d.doc_id}
                  className="rounded bg-red-900/60 px-1.5 py-0.5 text-[10px] text-red-300 transition hover:bg-red-800 hover:text-red-200 disabled:opacity-40"
                >
                  {deleting === d.doc_id ? "..." : "delete"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
