import React from "react";
import { EmptyState } from "../shared";

/**
 * List of items (title, type, excerpt) for the current section.
 * Used when we want a dedicated list view; CodexSidebar already embeds list per section.
 * This component can show a flat list of items for the selected section.
 */
export default function CodexDocumentList({ items = [], selectedId, onSelect, emptyMessage = "No items." }) {
  if (!items.length) {
    return <EmptyState message={emptyMessage} />;
  }
  return (
    <div className="space-y-1 overflow-auto">
      {items.map((item) => {
        const id = item.id ?? item.name ?? item.title;
        const title = item.title ?? item.name ?? "Untitled";
        const excerpt = item.summary ?? item.body ?? item.personality ?? "";
        const isSelected = selectedId === id;
        return (
          <button
            key={id}
            type="button"
            className={`w-full text-left border p-2 text-sm hover:border-[var(--gold)] ${isSelected ? "border-[var(--gold)] ring-1 ring-[var(--gold)]" : "border-[#5c3e23] bg-[#1a1008]"} text-[var(--text-1)]`}
            onClick={() => onSelect(item)}
          >
            <div className="font-heading">{title}</div>
            {excerpt && <div className="text-xs text-[var(--text-2)] line-clamp-2 mt-0.5">{excerpt}</div>}
          </button>
        );
      })}
    </div>
  );
}
