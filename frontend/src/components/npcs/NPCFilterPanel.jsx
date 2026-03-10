/**
 * Filters: faction, location, favorites. Codex-style labels and pill toggle.
 */
import React from "react";
import { Star } from "lucide-react";

export default function NPCFilterPanel({
  filterState,
  onFilterChange,
  factions = [],
  locations = [],
}) {
  const setFilter = (patch) => onFilterChange({ ...filterState, ...patch });

  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Filter
      </div>
      <label className="flex items-center gap-2 text-sm text-[var(--text-2)] cursor-pointer">
        <input
          type="checkbox"
          checked={!!filterState.favoritesOnly}
          onChange={(e) => setFilter({ favoritesOnly: e.target.checked })}
          className="rounded border-[#5c3e23] bg-[#1a1008] text-[var(--gold)] accent-[var(--gold)]"
        />
        <Star size={14} className="shrink-0" />
        <span>Favorites only</span>
      </label>
      {factions.length > 0 && (
        <label className="field-wrap">
          <span>Faction</span>
          <select
            className="chat-input w-full text-sm"
            value={filterState.faction || ""}
            onChange={(e) => setFilter({ faction: e.target.value || undefined })}
          >
            <option value="">All factions</option>
            {factions.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      )}
      {locations.length > 0 && (
        <label className="field-wrap">
          <span>Location</span>
          <select
            className="chat-input w-full text-sm"
            value={filterState.location || ""}
            onChange={(e) => setFilter({ location: e.target.value || undefined })}
          >
            <option value="">All locations</option>
            {locations.map((loc) => (
              <option key={loc} value={loc}>
                {loc}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
