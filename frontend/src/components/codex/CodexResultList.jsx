/**
 * Scrollable list of codex results. Uses CodexResultCard per item.
 */
import React from "react";
import CodexResultCard from "./CodexResultCard";
import { EmptyState } from "../shared";

export default function CodexResultList({ items = [], selectedItem, onSelectItem }) {
  if (!items.length) {
    return (
      <div className="flex-1 min-h-0 flex flex-col items-center justify-center p-4">
        <EmptyState message="No results. Try changing search or filters." />
      </div>
    );
  }
  return (
    <div className="flex flex-col min-h-0 flex-1">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2 shrink-0">
        Results ({items.length})
      </div>
      <div className="space-y-1.5 overflow-y-auto flex-1 min-h-0">
        {items.map((item) => (
          <CodexResultCard
            key={item.id}
            item={item}
            isSelected={selectedItem?.id === item.id}
            onClick={() => onSelectItem(item)}
          />
        ))}
      </div>
    </div>
  );
}
