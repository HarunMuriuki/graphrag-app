"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { GraphPayload } from "../lib/types";

// react-force-graph-2d touches `window` at import time (it wraps a canvas +
// d3-force simulation), so it must be loaded client-side only.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  Person: "#60a5fa", // blue-400
  Organization: "#fb923c", // orange-400
  Location: "#34d399", // emerald-400
  Product: "#c084fc", // purple-400
  Event: "#f472b6", // pink-400
  Concept: "#facc15", // yellow-400
  Date: "#94a3b8", // slate-400
};

function colorFor(type?: string | null): string {
  if (!type) return "#94a3b8";
  return TYPE_COLORS[type] || "#93c5fd";
}

export default function GraphPanel({ graph, title }: { graph: GraphPayload; title: string }) {
  const data = useMemo(
    () => ({
      nodes: graph.nodes.map((n) => ({ id: n.id, label: n.label, type: n.type })),
      links: graph.edges.map((e) => ({ source: e.source, target: e.target, label: e.label })),
    }),
    [graph]
  );

  if (graph.nodes.length === 0) {
    return (
      <div className="m-auto max-w-xs p-6 text-center text-sm text-slate-500">
        No graph data yet. Ingest a document and ask a question to see the entities and
        relationships used to answer it.
      </div>
    );
  }

  return (
    <div className="relative flex-1">
      <div className="absolute left-3 top-2 z-10 text-xs text-slate-500">
        {title} — {graph.nodes.length} entities, {graph.edges.length} relations
      </div>
      <ForceGraph2D
        graphData={data}
        nodeLabel={(n: any) => `${n.label} (${n.type || "Unknown"})`}
        nodeColor={(n: any) => colorFor(n.type)}
        linkLabel={(l: any) => l.label}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkColor={() => "rgba(148,163,184,0.5)"}
        backgroundColor="#0f172a"
        width={360}
        height={600}
      />
    </div>
  );
}
