"use client";

import { useRef, useState } from "react";
import { getJobStatus, uploadDocument } from "../lib/api";
import type { IngestJobStatus } from "../lib/types";

interface Props {
  onIngested?: () => void;
}

const STATUS_STYLES: Record<string, string> = {
  pending: "text-slate-400",
  processing: "text-blue-400",
  done: "text-emerald-400",
  error: "text-red-400",
};

export default function UploadPanel({ onIngested }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [jobs, setJobs] = useState<IngestJobStatus[]>([]);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      try {
        const { job_id } = await uploadDocument(file);
        pollJob(job_id, file.name);
      } catch (err) {
        setJobs((prev) => [
          {
            id: crypto.randomUUID(),
            filename: file.name,
            status: "error",
            total_chunks: 0,
            processed_chunks: 0,
            error: err instanceof Error ? err.message : "Upload failed",
          },
          ...prev,
        ]);
      }
    }
  }

  function pollJob(jobId: string, filename: string) {
    setJobs((prev) => [
      { id: jobId, filename, status: "pending", total_chunks: 0, processed_chunks: 0 },
      ...prev,
    ]);

    const interval = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        setJobs((prev) => prev.map((j) => (j.id === jobId ? status : j)));
        if (status.status === "done" || status.status === "error") {
          clearInterval(interval);
          if (status.status === "done") onIngested?.();
        }
      } catch {
        clearInterval(interval);
      }
    }, 1500);
  }

  return (
    <div>
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Upload documents
      </p>
      <label className="block cursor-pointer rounded-lg border-2 border-dashed border-slate-700 p-4 text-center text-xs text-slate-400 transition-colors hover:border-blue-500 hover:text-slate-300">
        <input
          ref={inputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <span className="block font-medium text-slate-300">Drop or click to upload</span>
        <span className="mt-1 block">PDF, DOCX, TXT, MD, and other text files</span>
      </label>

      {jobs.length > 0 && (
        <div className="mt-3 flex flex-col gap-1.5">
          {jobs.map((job) => (
            <div key={job.id} className="rounded-md bg-slate-800 px-3 py-1.5 text-xs">
              <div className="truncate text-slate-300">{job.filename}</div>
              <div className={STATUS_STYLES[job.status] ?? "text-slate-400"}>
                {job.status === "processing"
                  ? `processing (${job.processed_chunks}/${job.total_chunks || "?"} chunks)`
                  : job.status === "error"
                  ? job.error || "error"
                  : job.status}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
