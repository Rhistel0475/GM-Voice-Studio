import { useMemo, useState } from "react";
import { useAppState } from "../context/AppStateContext";
import { buildCodexIntelligence } from "../lib/codexIntelligence";
import { CODEX_TABS } from "../components/live-board/constants";
import { EmptyState, FantasyButton } from "../components/shared";
import { BookOpen, Upload, Search } from "lucide-react";

/**
 * PrepPage — 3-column campaign management interface.
 *
 * LEFT  (250px): search + category filters + action buttons
 * CENTER (flex): codex content list, or injected prep/library UI
 * RIGHT  (320px): selected entry detail with actions
 *
 * prepContent / libraryContent are injected from App.jsx because
 * PrepRoom and AdventureIntake are still defined inline there.
 */
export default function PrepPage({
  prepContent,
  libraryContent,
  onNavigate,
}) {
  const { campaignData } = useAppState();
  const [activeCategory, setActiveCategory] = useState("npcs");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [centerMode, setCenterMode] = useState("codex"); // "codex" | "upload" | "prep"

  const intelligence = useMemo(
    () => buildCodexIntelligence(campaignData),
    [campaignData]
  );

  const entries = useMemo(() => {
    const all = intelligence[activeCategory] || [];
    if (!search.trim()) return all;
    const q = search.toLowerCase();
    return all.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        (e.description || "").toLowerCase().includes(q)
    );
  }, [intelligence, activeCategory, search]);

  const handleCategoryChange = (cat) => {
    setActiveCategory(cat);
    setSelected(null);
    setCenterMode("codex");
  };

  const handleOpenMode = (mode) => {
    setSelected(null);
    setCenterMode(mode);
  };

  const handleEntryClick = (entry) => {
    setSelected((prev) => (prev?.id === entry.id ? null : entry));
  };

  const totalForCategory = intelligence[activeCategory]?.length ?? 0;

  return (
    <div className="flex h-full min-h-0 overflow-hidden gap-0">

      {/* ── LEFT: Filters + Actions ────────────────────────────────── */}
      <aside
        className="flex-shrink-0 flex flex-col gap-3 p-3 border-r border-[#3a2510] bg-[#12090400] overflow-y-auto"
        style={{ width: "220px" }}
      >
        {/* Search */}
        <div className="relative">
          <Search
            size={13}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-2)] pointer-events-none"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setCenterMode("codex");
            }}
            placeholder="Search…"
            className="w-full rounded-md border border-[#4a3018] bg-[#1c1008] pl-7 pr-3 py-1.5 text-xs text-[var(--text-1)] placeholder:text-[var(--text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
          />
        </div>

        {/* Category filters */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1 px-1">
            Categories
          </div>
          {CODEX_TABS.map((tab) => {
            const count = intelligence[tab.key]?.length ?? 0;
            const isActive = activeCategory === tab.key && centerMode === "codex";
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => handleCategoryChange(tab.key)}
                className={[
                  "flex items-center justify-between rounded-md px-2.5 py-1.5 text-xs text-left transition-colors",
                  isActive
                    ? "bg-[#2e1c08] border border-[var(--gold)] text-[var(--gold)]"
                    : "border border-transparent text-[var(--text-1)] hover:bg-[#1e1208] hover:border-[#4a3018]",
                ].join(" ")}
              >
                <span>{tab.label}</span>
                {count > 0 && (
                  <span
                    className={[
                      "text-[10px] rounded-full px-1.5 py-0.5 min-w-[20px] text-center",
                      isActive
                        ? "bg-[var(--gold)] text-[#1a0e04]"
                        : "bg-[#2a1a0a] text-[var(--text-2)]",
                    ].join(" ")}
                  >
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-1.5 mt-auto pt-3 border-t border-[#3a2510]">
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1 px-1">
            Actions
          </div>
          <button
            type="button"
            onClick={() => handleOpenMode("upload")}
            className={[
              "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-left transition-colors",
              centerMode === "upload"
                ? "bg-[#2e1c08] border border-[var(--gold)] text-[var(--gold)]"
                : "border border-[#4a3018] text-[var(--text-1)] hover:bg-[#1e1208]",
            ].join(" ")}
          >
            <Upload size={13} className="shrink-0" />
            Upload Adventure
          </button>
          <button
            type="button"
            onClick={() => handleOpenMode("prep")}
            className={[
              "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-left transition-colors",
              centerMode === "prep"
                ? "bg-[#2e1c08] border border-[var(--gold)] text-[var(--gold)]"
                : "border border-[#4a3018] text-[var(--text-1)] hover:bg-[#1e1208]",
            ].join(" ")}
          >
            <BookOpen size={13} className="shrink-0" />
            Scene Builder
          </button>
        </div>
      </aside>

      {/* ── CENTER: Content list or injected view ──────────────────── */}
      <main className="flex-1 min-h-0 min-w-0 flex flex-col overflow-hidden border-r border-[#3a2510]">
        {centerMode === "upload" ? (
          <div className="flex-1 min-h-0 overflow-y-auto p-4">
            {libraryContent ?? (
              <EmptyState message="Upload content not available." />
            )}
          </div>
        ) : centerMode === "prep" ? (
          <div className="flex-1 min-h-0 overflow-y-auto p-4">
            {prepContent ?? (
              <EmptyState message="Scene builder not available." />
            )}
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#3a2510] flex-shrink-0">
              <div className="font-heading text-sm text-[var(--gold)]">
                {CODEX_TABS.find((t) => t.key === activeCategory)?.label ?? "Codex"}
              </div>
              <div className="text-xs text-[var(--text-2)]">
                {entries.length !== totalForCategory
                  ? `${entries.length} of ${totalForCategory}`
                  : `${totalForCategory} entr${totalForCategory !== 1 ? "ies" : "y"}`}
              </div>
            </div>

            {/* Entry list */}
            <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-1">
              {entries.length === 0 ? (
                <div className="p-4">
                  <EmptyState
                    message={
                      search
                        ? `No matches for "${search}".`
                        : `No ${activeCategory} in this campaign yet.`
                    }
                  />
                </div>
              ) : (
                entries.map((entry) => {
                  const isSelected = selected?.id === entry.id;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      onClick={() => handleEntryClick(entry)}
                      className={[
                        "w-full text-left rounded-md px-3 py-2 border transition-all",
                        isSelected
                          ? "bg-[#221808] border-[var(--gold)] shadow-[0_0_8px_rgba(202,167,75,0.18)]"
                          : "bg-[#1a1008]/90 border-[#5c3e23] hover:border-[var(--gold)] hover:bg-[#221808]",
                      ].join(" ")}
                    >
                      <div className="font-heading text-sm text-[var(--text-1)] truncate">
                        {entry.title}
                      </div>
                      {entry.subtitle && (
                        <div className="text-[11px] text-[var(--gold)] truncate">
                          {entry.subtitle}
                        </div>
                      )}
                      {entry.description && (
                        <div className="text-[11px] text-[var(--text-2)] line-clamp-2 mt-0.5">
                          {entry.description}
                        </div>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </>
        )}
      </main>

      {/* ── RIGHT: Detail panel ────────────────────────────────────── */}
      <aside
        className="flex-shrink-0 flex flex-col min-h-0 overflow-hidden"
        style={{ width: "300px" }}
      >
        {selected ? (
          <DetailPanel
            entry={selected}
            onNavigate={onNavigate}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center p-6">
            <p className="text-xs text-[var(--text-2)] text-center leading-relaxed">
              Select an entry to view details, relationships, and actions.
            </p>
          </div>
        )}
      </aside>

    </div>
  );
}

/* ─── Detail Panel ────────────────────────────────────────────────── */

function DetailPanel({ entry, onNavigate }) {
  return (
    <div className="flex flex-col h-full min-h-0 overflow-y-auto p-3 gap-3">

      {/* Header */}
      <div className="border-b border-[#3a2510] pb-3">
        <div className="font-heading text-[var(--gold)] text-base leading-tight">
          {entry.title}
        </div>
        {entry.subtitle && (
          <div className="text-xs text-[#d4b36a] mt-0.5">{entry.subtitle}</div>
        )}
        <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mt-1">
          {entry.tab}
        </div>
      </div>

      {/* Description */}
      {entry.description && (
        <div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1.5">
            Description
          </div>
          <p className="text-xs text-[var(--text-1)] leading-relaxed whitespace-pre-wrap">
            {entry.description}
          </p>
        </div>
      )}

      {/* Relationships */}
      {entry.relationships?.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1.5">
            Related Entities
          </div>
          <div className="flex flex-wrap gap-1.5">
            {entry.relationships.map((rel) => (
              <span
                key={rel}
                className="rounded-full border border-[#6b4a28] bg-[#24170b] px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] text-[var(--text-2)]"
              >
                {rel}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="mt-auto pt-3 border-t border-[#3a2510] flex flex-col gap-2">
        <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-0.5">
          Actions
        </div>
        <div className="flex flex-wrap gap-2">
          <FantasyButton
            variant="secondary"
            className="text-xs"
            onClick={() => onNavigate?.("voice-studio")}
          >
            Assign Voice
          </FantasyButton>
          <FantasyButton
            variant="ghost"
            className="text-xs"
            onClick={() => onNavigate?.("live")}
          >
            Open in Live
          </FantasyButton>
        </div>
      </div>
    </div>
  );
}
