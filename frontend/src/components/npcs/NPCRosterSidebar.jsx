import React from "react";
import NPCSearchInput from "./NPCSearchInput";
import NPCFilterPanel from "./NPCFilterPanel";
import NPCList from "./NPCList";

/**
 * Left panel: search, filters, and list. Matches CodexSidebar structure (plaque + labels + content).
 */
export default function NPCRosterSidebar({
  filteredNpcs = [],
  filterState,
  onFilterChange,
  selectedNpc,
  onSelectNpc,
  factions = [],
  locations = [],
}) {
  return (
    <div className="flex flex-col gap-3 min-h-0 h-full overflow-hidden">
      <div className="plaque mb-1 shrink-0">NPC Roster</div>
      <div className="flex flex-col gap-1 shrink-0">
        <label className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
          Search
        </label>
        <NPCSearchInput
          value={filterState.query}
          onChange={(query) => onFilterChange({ ...filterState, query })}
        />
      </div>
      <div className="shrink-0">
        <NPCFilterPanel
          filterState={filterState}
          onFilterChange={onFilterChange}
          factions={factions}
          locations={locations}
        />
      </div>
      <div className="min-h-0 flex-1 flex flex-col overflow-hidden">
        <NPCList
          filteredNpcs={filteredNpcs}
          selectedNpc={selectedNpc}
          onSelectNpc={onSelectNpc}
        />
      </div>
    </div>
  );
}
