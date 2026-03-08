import React from "react";

/**
 * Single roster row: name, role pill, excerpt. Matches CodexResultCard selected/ring style for scanability.
 */
export default function NPCListItem({ npc, selected, onSelect }) {
  const roleLabel = npc.role || npc.profession || "NPC";
  const excerpt = npc.summary || npc.notes || "";

  return (
    <button
      type="button"
      className={`w-full text-left border p-2 rounded transition-colors ${
        selected
          ? "border-[var(--gold)] ring-1 ring-[var(--gold)] bg-[#1a1008]"
          : "border-[#5c3e23] bg-[#1a1008] hover:border-[var(--gold)]"
      }`}
      onClick={() => onSelect(npc)}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-heading text-[var(--text-1)] text-sm">
          {npc.name}
        </span>
        <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border border-[#5c3e23] text-[var(--text-2)] shrink-0">
          {roleLabel}
        </span>
      </div>
      {excerpt && (
        <div className="text-xs text-[var(--text-2)] line-clamp-2 mt-1">
          {excerpt}
        </div>
      )}
    </button>
  );
}
