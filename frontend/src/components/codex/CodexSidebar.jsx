import CodexSearchBar from "./CodexSearchBar";
import CodexFilterPanel from "./CodexFilterPanel";

/**
 * Left sidebar: campaign selector, search bar, filter panel (category + tags).
 * Used by CodexScreen for the research view.
 */
export default function CodexSidebar({
  campaignData,
  campaigns = [],
  onCampaignSelect,
  filterState,
  onFilterChange,
  availableTags = [],
}) {
  const currentCampaignName = campaignData?.title || "No campaign loaded";

  return (
    <div className="flex flex-col gap-3 overflow-auto h-full min-h-0">
      <div className="plaque mb-1 shrink-0">Research</div>
      <div className="flex flex-col gap-1 shrink-0">
        <label className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
          Campaign
        </label>
        <select
          className="chat-input w-full"
          value={campaignData?.id ?? ""}
          onChange={(e) => {
            const val = e.target.value;
            const id = val === "" ? null : Number(val);
            onCampaignSelect?.(id);
          }}
          aria-label="Select campaign"
        >
          <option value="">Current: {currentCampaignName}</option>
          {campaigns.map((c) => (
            <option key={c.id} value={c.id}>
              {c.title || `Campaign #${c.id}`}
            </option>
          ))}
        </select>
        {campaigns.length === 0 && (
          <p className="text-xs text-[var(--text-2)]">Load campaigns from Library.</p>
        )}
      </div>
      <div className="flex flex-col gap-1 shrink-0">
        <label className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
          Search
        </label>
        <CodexSearchBar
          value={filterState.query || ""}
          onChange={(q) => onFilterChange((prev) => ({ ...prev, query: q }))}
          placeholder="Search codex…"
        />
      </div>
      <div className="shrink-0">
        <CodexFilterPanel
          filterState={filterState}
          onFilterChange={onFilterChange}
          availableTags={availableTags}
        />
      </div>
    </div>
  );
}
