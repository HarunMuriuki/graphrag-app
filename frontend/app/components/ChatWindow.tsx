"use client";

import { useEffect, useRef, useState } from "react";
import { streamChat } from "../lib/api";
import type { ChatMessage, GraphPayload } from "../lib/types";
import MessageBubble from "./MessageBubble";

interface Props {
  onGraphUpdate: (graph: GraphPayload) => void;
}

export default function ChatWindow({ onGraphUpdate }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function send() {
    const question = input.trim();
    if (!question || busy) return;

    setError(null);
    setInput("");
    setBusy(true);

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: question };
    const assistantId = crypto.randomUUID();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await streamChat(question, {
      onContext: (ctx) => {
        onGraphUpdate(ctx.graph);
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, sources: ctx.sources } : m))
        );
      },
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + token } : m))
        );
      },
      onDone: () => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m))
        );
        setBusy(false);
      },
      onError: (message) => {
        setError(message);
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, isStreaming: false } : m))
        );
        setBusy(false);
      },
    });
  }

  return (
    <div className="flex h-screen flex-col">
      <div ref={scrollRef} className="flex flex-1 flex-col gap-4 overflow-y-auto p-6">
        {messages.length === 0 && (
          <div className="m-auto max-w-md text-center text-sm text-slate-500">
            Upload a document on the left, then ask a question. Answers combine semantic
            search over your documents with facts pulled from the knowledge graph built
            during ingestion — shown on the right as they&apos;re used.
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {error && (
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      <div className="flex gap-3 border-t border-slate-800 p-4">
        <input
          value={input}
          placeholder="Ask a question about your documents..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
          }}
          disabled={busy}
          className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-60"
        />
        <button
          onClick={send}
          disabled={busy || !input.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-blue-900/60 disabled:text-slate-400"
        >
          {busy ? "Thinking…" : "Send"}
        </button>
      </div>
    </div>
  );
}
