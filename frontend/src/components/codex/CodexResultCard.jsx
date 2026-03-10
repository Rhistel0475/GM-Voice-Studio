import React from "react";
import { getDocumentPlaceholder, getLocationPlaceholder, getNpcPlaceholder } from "../../lib/placeholders";

const CATEGORY_LABELS = {
  document: "Doc",
  npc: "NPC",
  location: "Location",
  rule: "Rule",
  lore: "Lore",
  faction: "Faction",
};

/**
 * Single result row: title, category pill, excerpt. Research-desk style; selected ring.
 */
export default function CodexResultCard({ item, isSelected, onClick }) {
  const label = CATEGORY_LABELS[item.category] || item.category;
  const excerpt = item.excerpt || item.summary || "";
  const category = item.category;
  const iconSrc =
    category === "location"
      ? getLocationPlaceholder()
      : category === "npc"
      ? getNpcPlaceholder(item.tags?.[0])
      : getDocumentPlaceholder(category);

  return (
    <button
      type="button"
      className={`w-full text-left rounded transition-colors border-l-4 ${
        isSelected
          ? "border-l-[var(--gold)] border border-[var(--gold)] ring-1 ring-[var(--gold)] bg-[#1a1008]"
          : "border-l-transparent border border-[#5c3e23] bg-[#1a1008] hover:border-[var(--gold)] hover:border-l-[var(--gold)]/60"
      } p-2.5`}
      onClick={onClick}
    >
      <div className="flex items-center gap-2 flex-wrap">
        {iconSrc && (
          <span className="w-6 h-6 rounded-full overflow-hidden border border-[#5c3e23] bg-[#1a1008] flex items-center justify-center shrink-0">
            <img src={iconSrc} alt="" className="w-full h-full object-cover" />
          </span>
        )}
        <span className="font-heading text-[var(--text-1)] text-sm leading-tight">
          {item.title}
        </span>
        <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border border-[#5c3e23] text-[var(--text-2)] shrink-0">
          {label}
        </span>
      </div>
      {excerpt && (
        <div className="text-xs text-[var(--text-2)] line-clamp-2 mt-1.5 leading-snug">
          {excerpt}
        </div>
      )}
    </button>
  );
}
