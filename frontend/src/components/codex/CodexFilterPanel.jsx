import React from "react";
import { CODEX_CATEGORIES } from "../../types/codex";
import { FantasyButton } from "../shared";

const CATEGORY_LABELS = {
  all: "All",
  document: "Documents",
  npc: "NPCs",
  location: "Locations",
  rule: "Rules",
  lore: "Lore",
  faction: "Factions",
};

/**
 * Category and optional tag filter for the Codex. Category chips; tags as multi-select or pills.
 */
export default function CodexFilterPanel({
  filterState,
  onFilterChange,
  availableTags = [],
}) {
  const setCategory = (category) => {
    onFilterChange((prev) => ({ ...prev, category }));
  };

  const toggleTag = (tag) => {
    onFilterChange((prev) => {
      const tags = prev.tags || [];
      const next = tags.includes(tag) ? tags.filter((t) => t !== tag) : [...tags, tag];
      return { ...prev, tags: next };
    });
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Category
      </div>
      <div className="flex flex-wrap gap-1">
        <FantasyButton
          variant={filterState.category === "all" ? "primary" : "ghost"}
          className="text-xs"
          onClick={() => setCategory("all")}
        >
          All
        </FantasyButton>
        {CODEX_CATEGORIES.map((cat) => (
          <FantasyButton
            key={cat}
            variant={filterState.category === cat ? "primary" : "ghost"}
            className="text-xs"
            onClick={() => setCategory(cat)}
          >
            {CATEGORY_LABELS[cat] || cat}
          </FantasyButton>
        ))}
      </div>
      {availableTags.length > 0 && (
        <>
          <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mt-2">
            Tags
          </div>
          <div className="flex flex-wrap gap-1.5">
            {availableTags.slice(0, 12).map((tag) => {
              const active = (filterState.tags || []).includes(tag);
              return (
                <button
                  key={tag}
                  type="button"
                  className={`rounded border px-2 py-0.5 text-xs transition-colors ${
                    active
                      ? "border-[var(--gold)] bg-[rgba(202,167,75,0.18)] text-[var(--candle-glow)]"
                      : "border-[#5c3e23] bg-[#1a1008] text-[var(--text-2)] hover:border-[var(--gold)]"
                  }`}
                  onClick={() => toggleTag(tag)}
                >
                  {tag}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
