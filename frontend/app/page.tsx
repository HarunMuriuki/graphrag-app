"use client";

import { useCallback, useEffect, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import DocumentList from "./components/DocumentList";
import GraphPanel from "./components/GraphPanel";
import UploadPanel from "./components/UploadPanel";
import { getGraphOverview } from "./lib/api";
import type { GraphPayload } from "./lib/types";

const EMPTY_GRAPH: GraphPayload = { nodes: [], edges: [] };

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [graph, setGraph] = useState<GraphPayload>(EMPTY_GRAPH);
  const [graphTitle, setGraphTitle] = useState("Knowledge graph overview");

  const loadOverview = useCallback(() => {
    getGraphOverview()
      .then((g) => {
        setGraph(g);
        setGraphTitle("Knowledge graph overview");
      })
      .catch(() => setGraph(EMPTY_GRAPH));
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview, refreshKey]);

  return (
    <div className="grid h-screen grid-cols-[300px_1fr_380px]">
      <aside className="flex flex-col gap-6 overflow-y-auto border-r border-slate-800 bg-slate-900 p-4">
        <div>
          <h1 className="text-lg font-semibold text-slate-50">GraphRAG</h1>
          <p className="mt-1 text-xs text-slate-400">
            Hybrid vector + knowledge-graph retrieval, running fully locally.
          </p>
        </div>
        <UploadPanel onIngested={() => setRefreshKey((k) => k + 1)} />
        <DocumentList refreshKey={refreshKey} onDeleted={() => setRefreshKey((k) => k + 1)} />
      </aside>

      <ChatWindow
        onGraphUpdate={(g) => {
          setGraph(g);
          setGraphTitle("Context used for this answer");
        }}
      />

      <aside className="flex flex-col border-l border-slate-800 bg-slate-900">
        <div className="border-b border-slate-800 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Knowledge graph
          </p>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-500">
            Colors indicate entity type. This panel shows the whole graph until you ask a
            question, then switches to just the entities/relations used in that answer.
          </p>
        </div>
        <GraphPanel graph={graph} title={graphTitle} />
      </aside>
    </div>
  );
}
