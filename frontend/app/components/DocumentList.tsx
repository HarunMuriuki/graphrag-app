"use client";

import { useEffect, useState } from "react";
import { listDocuments } from "../lib/api";
import type { DocumentSummary } from "../lib/types";

interface Props {
  refreshKey: number;
}

export default function DocumentList({ refreshKey }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch(() => setDocs([]));
  }, [refreshKey]);

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
              <span className="shrink-0 text-slate-500">{d.chunks} chunks</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
