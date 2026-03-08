import React from "react";

/**
 * Step 2: Training progress bar and status label. Fantasy bar; optional pulse when processing.
 */
export default function VoiceTrainingStatus({ status, progress, stepLabel }) {
  const statusText = stepLabel || status || "Processing…";
  const pct = typeof progress === "number" ? Math.min(100, Math.max(0, progress)) : 0;
  const isActive = status === "processing" || status === "queued" || !status;

  return (
    <div className="flex flex-col gap-3">
      <div className="w-full h-3 rounded overflow-hidden border border-[#5c3e23] bg-[#1a1008]">
        <div
          className={`h-full transition-all duration-300 ease-out ${
            isActive ? "voice-training-pulse" : ""
          }`}
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #8c6435, var(--gold))",
            boxShadow: "inset 0 0 8px rgba(255, 210, 122, 0.2)",
          }}
        />
      </div>
      <p className="font-heading text-[var(--ink-1)] text-sm">
        {statusText}
      </p>
      <p className="text-xs text-[var(--text-2)] opacity-80">{pct}%</p>
    </div>
  );
}
