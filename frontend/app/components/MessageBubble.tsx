import type { ChatMessage } from "../lib/types";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[70%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white"
            : "border border-slate-700 bg-slate-800 text-slate-100"
        }`}
      >
        <div>
          {message.content}
          {message.isStreaming && <span className="animate-pulse">▍</span>}
        </div>

        {message.sources && message.sources.length > 0 && (
          <details className="mt-2 border-t border-slate-700/70 pt-2 text-xs text-slate-400">
            <summary className="cursor-pointer select-none hover:text-slate-300">
              {message.sources.length} source passage(s)
            </summary>
            <div className="mt-1.5 flex flex-col gap-1.5">
              {message.sources.map((s) => (
                <div key={s.chunk_id} className="rounded-md bg-slate-900/60 p-2">
                  <div className="font-medium text-blue-400">
                    {s.source} · score {s.score.toFixed(3)}
                  </div>
                  <div className="mt-0.5 text-slate-400">
                    {s.text.slice(0, 240)}
                    {s.text.length > 240 ? "…" : ""}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
