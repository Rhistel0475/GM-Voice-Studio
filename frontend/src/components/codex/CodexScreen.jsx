import React, { useState, useMemo, useEffect } from "react";
import { defaultCodexFilterState } from "../../types/codex";
import { getCodexItems } from "../../lib/api/codex";
import CodexSidebar from "./CodexSidebar";
import CodexResultList from "./CodexResultList";
import CodexDetailPanel from "./CodexDetailPanel";

/**
 * Filter items by query, category, and tags.
 */
function filterItems(items, filterState) {
  let out = items;
  const q = (filterState.query || "").trim().toLowerCase();
  if (q) {
    out = out.filter(
      (item) =>
        (item.title && item.title.toLowerCase().includes(q)) ||
        (item.summary && item.summary.toLowerCase().includes(q)) ||
        (item.tags && item.tags.some((t) => t.toLowerCase().includes(q)))
    );
  }
  if (filterState.category && filterState.category !== "all") {
    out = out.filter((item) => item.category === filterState.category);
  }
  if (filterState.tags && filterState.tags.length > 0) {
    const tagSet = new Set(filterState.tags.map((t) => t.toLowerCase()));
    out = out.filter(
      (item) => item.tags && item.tags.some((t) => tagSet.has(t.toLowerCase()))
    );
  }
  return out;
}

export default function CodexScreen({ campaignData, authFetch, campaigns = [], onCampaignSelect, onAddToLiveBoard }) {
  const [filterState, setFilterState] = useState(defaultCodexFilterState());
  const [selectedItem, setSelectedItem] = useState(null);

  const items = useMemo(
    () => getCodexItems(campaignData, authFetch),
    [campaignData, authFetch]
  );

  const filteredItems = useMemo(
    () => filterItems(items, filterState),
    [items, filterState]
  );

  const availableTags = useMemo(() => {
    const set = new Set();
    items.forEach((item) => {
      (item.tags || []).forEach((t) => set.add(t));
    });
    return Array.from(set);
  }, [items]);

  return (
    <section className="codex-screen min-h-0 flex-1 grid grid-cols-1 md:grid-cols-12 gap-3 p-2 md:p-3 bg-[var(--wood-2)]/50">
      {/* Left: research navigation */}
      <div className="md:col-span-3 lg:col-span-3 xl:col-span-3 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <CodexSidebar
          campaignData={campaignData}
          campaigns={campaigns}
          onCampaignSelect={onCampaignSelect}
          filterState={filterState}
          onFilterChange={setFilterState}
          availableTags={availableTags}
          onSelectItem={setSelectedItem}
        />
      </div>

      {/* Middle: result list */}
      <div className="md:col-span-4 lg:col-span-4 xl:col-span-4 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <CodexResultList
          items={filteredItems}
          selectedItem={selectedItem}
          onSelectItem={setSelectedItem}
        />
      </div>

      {/* Right: detail panel */}
      <div className="md:col-span-5 lg:col-span-5 xl:col-span-5 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2 overflow-hidden min-w-0">
        <CodexDetailPanel
          item={selectedItem}
          authFetch={authFetch}
          onAddToLiveBoard={onAddToLiveBoard}
        />
      </div>
    </section>
  );
}
