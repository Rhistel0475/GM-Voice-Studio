import React from "react";

const statusStyles = {
  ready: "border-[#5f7d63] text-[#9ec24f] bg-[rgba(94,124,99,0.2)]",
  recording: "border-[#c05050] text-[#ff6b6b] bg-[rgba(192,80,80,0.2)]",
  generating: "border-[var(--gold)] text-[var(--candle-glow)] bg-[rgba(202,167,75,0.15)]",
  playing: "border-[#5f7d63] text-[#7ec24f] bg-[rgba(94,124,99,0.2)]",
  offline: "border-[#6b5230] text-[#9c7a3a] bg-[rgba(107,82,48,0.2)]",
  saved: "border-[#5f7d63] text-[#7ec24f] bg-[rgba(94,124,99,0.15)]",
  training: "border-[var(--gold)] text-[var(--candle-glow)] bg-[rgba(202,167,75,0.15)]",
  failed: "border-[#c05050] text-[#ff6b6b] bg-[rgba(192,80,80,0.2)]",
};

/**
 * Small pill for status: Ready | Recording | Generating | Offline | Saved.
 */
export default function StatusPill({ status = "offline", className = "" }) {
  const style = statusStyles[status] ?? statusStyles.offline;
  const label = status.charAt(0).toUpperCase() + (status.slice(1) || "Offline");
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${style} ${className}`.trim()}
      role="status"
    >
      {label}
    </span>
  );
}
