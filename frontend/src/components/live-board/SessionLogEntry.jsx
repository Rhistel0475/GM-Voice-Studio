import React from "react";

const roleStyles = {
  player: "border-[#5d472a] bg-[#1c120a] text-[#e6c785]",
  error: "border-red-900/70 bg-red-950/20 text-red-300",
  stat_block: "border-[#c79f5b] bg-[#0e1a0e] text-[#ffe08a]",
  lore: "border-[#7a5a30] bg-[#eddcb8]/10 text-[#c8a97a]",
};
const roleLabels = {
  player: "Table",
  error: "System",
  stat_block: "Stat Block",
  lore: "Lore",
};

export default function SessionLogEntry({ entry }) {
  const style = roleStyles[entry.role] ?? "border-[#37553e] bg-[#102016] text-[#d4f0cf]";
  const label = roleLabels[entry.role] ?? "Co-DM";
  return (
    <div className={`rounded border px-2 py-1 text-xs ${style}`}>
      <div className="mb-0.5 flex items-center justify-between">
        <span className="font-heading text-[10px] tracking-wide uppercase">{label}</span>
        {entry.meta && <span className="text-[10px] opacity-80">{entry.meta}</span>}
      </div>
      {entry.role === "stat_block" ? (
        <pre className="whitespace-pre-wrap font-mono text-xs">{entry.text}</pre>
      ) : (
        <div className="whitespace-pre-wrap">{entry.text}</div>
      )}
    </div>
  );
}
