import { useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { useAppState } from "../context/AppStateContext";
import { buildCodexIntelligence } from "../lib/codexIntelligence";
import { CODEX_TABS } from "../components/live-board/constants";
import { EmptyState, FantasyButton } from "../components/shared";
import { BookOpen, Upload, Search, Scroll, X } from "lucide-react";

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
  const { search: locationSearch } = useLocation();
  const [activeCategory, setActiveCategory] = useState("npcs");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(null);
  const [centerMode, setCenterMode] = useState(() => {
    const mode = new URLSearchParams(locationSearch).get("mode");
    return mode === "upload" || mode === "prep" ? mode : "codex";
  }); // "codex" | "upload" | "prep"

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
        className="flex-shrink-0 flex flex-col gap-0 border-r border-[#3a2510] overflow-y-auto bg-[#0d0804]"
        style={{ width: "210px" }}
      >
        {/* Search */}
        <div className="p-3 pb-4 border-b border-[#2a1a0a]">
          <div className="sidebar-section-label">Search</div>
          <div className="relative mt-1">
            <Search
              size={12}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-2)] pointer-events-none"
            />
            <input
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setCenterMode("codex");
              }}
              placeholder="Filter entries…"
              className="w-full rounded-md border border-[#4a3018] bg-[#1c1008] pl-7 pr-3 py-1.5 text-xs text-[var(--text-1)] placeholder:text-[var(--text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
            />
          </div>
        </div>

        {/* Category filters */}
        <div className="p-3 pb-4 flex-1 border-b border-[#2a1a0a]">
          <div className="sidebar-section-label">Categories</div>
          <div className="flex flex-col gap-0.5 mt-2">
            {CODEX_TABS.map((tab) => {
              const count = intelligence[tab.key]?.length ?? 0;
              const isActive = activeCategory === tab.key && centerMode === "codex";
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => handleCategoryChange(tab.key)}
                  className={[
                    "flex items-center justify-between rounded px-2.5 py-2 text-xs text-left transition-all",
                    isActive
                      ? "bg-[#2a1608] border border-[var(--gold)]/60 text-[var(--gold)] font-semibold shadow-[inset_0_0_8px_rgba(202,167,75,0.08)]"
                      : "border border-transparent text-[var(--text-2)] hover:bg-[#181008] hover:text-[var(--text-1)] hover:border-[#3a2510]",
                  ].join(" ")}
                >
                  <span>{tab.label}</span>
                  {count > 0 && (
                    <span
                      className={[
                        "text-[9px] rounded-full px-1.5 py-0.5 min-w-[18px] text-center tabular-nums",
                        isActive
                          ? "bg-[var(--gold)] text-[#1a0e04] font-bold"
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
        </div>

        {/* Action buttons */}
        <div className="p-3 pb-4 flex flex-col gap-1.5">
          <div className="sidebar-section-label">Tools</div>
          <button
            type="button"
            onClick={() => handleOpenMode("upload")}
            className={[
              "flex items-center gap-2 rounded px-2.5 py-2 text-xs text-left transition-colors",
              centerMode === "upload"
                ? "bg-[#2e1c08] border border-[var(--gold)] text-[var(--gold)]"
                : "border border-[#4a3018] text-[var(--text-1)] hover:bg-[#1e1208]",
            ].join(" ")}
          >
            <Upload size={12} className="shrink-0" />
            Upload Adventure
          </button>
          <button
            type="button"
            onClick={() => handleOpenMode("prep")}
            className={[
              "flex items-center gap-2 rounded px-2.5 py-2 text-xs text-left transition-colors",
              centerMode === "prep"
                ? "bg-[#2e1c08] border border-[var(--gold)] text-[var(--gold)]"
                : "border border-[#4a3018] text-[var(--text-1)] hover:bg-[#1e1208]",
            ].join(" ")}
          >
            <BookOpen size={12} className="shrink-0" />
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
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#3a2510] flex-shrink-0 bg-[#150e07]">
              <div className="font-heading text-sm text-[var(--gold)] tracking-wide uppercase">
                {CODEX_TABS.find((t) => t.key === activeCategory)?.label ?? "Codex"}
              </div>
              <div className="text-[10px] text-[var(--text-2)] tabular-nums">
                {entries.length !== totalForCategory
                  ? `${entries.length} / ${totalForCategory}`
                  : `${totalForCategory} entr${totalForCategory !== 1 ? "ies" : "y"}`}
              </div>
            </div>

            {/* Entry list */}
            <div className="flex-1 min-h-0 overflow-y-auto px-2 py-2 space-y-0.5">
              {entries.length === 0 ? (
                search ? (
                  /* Search no-results */
                  <div className="flex flex-col items-center gap-3 mt-10 px-6 text-center">
                    <Search size={28} className="text-[var(--text-2)] opacity-40" />
                    <div>
                      <p className="font-heading text-sm text-[var(--text-1)] tracking-wide">No Matches Found</p>
                      <p className="text-xs text-[var(--text-2)] mt-1 leading-relaxed">
                        No entries match <span className="text-[var(--gold)]">"{search}"</span>. Try a different term.
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setSearch("")}
                      className="flex items-center gap-1.5 text-xs text-[var(--text-2)] hover:text-[var(--gold)] transition-colors"
                    >
                      <X size={11} /> Clear search
                    </button>
                  </div>
                ) : (
                  /* Category empty */
                  <div className="flex flex-col items-center gap-4 mt-8 mx-2 px-5 py-7 rounded-lg border border-[#3a2510] bg-[#0f0804]/60 text-center">
                    <Scroll size={30} className="text-[var(--gold)] opacity-35" />
                    <div>
                      <p className="font-heading text-sm text-[var(--text-1)] tracking-wide capitalize">
                        No {activeCategory} in Your Codex
                      </p>
                      <p className="text-xs text-[var(--text-2)] mt-1.5 leading-relaxed max-w-[200px] mx-auto">
                        Upload an adventure module or build scenes manually to populate this section.
                      </p>
                    </div>
                    <div className="flex flex-col gap-2 w-full">
                      <button
                        type="button"
                        onClick={() => handleOpenMode("upload")}
                        className="flex items-center justify-center gap-1.5 w-full rounded px-3 py-2 text-xs border border-[var(--gold)] text-[var(--gold)] bg-[#1e1208] hover:bg-[#2a1a0a] transition-colors font-heading tracking-wide"
                      >
                        <Upload size={11} /> Upload Adventure
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOpenMode("prep")}
                        className="flex items-center justify-center gap-1.5 w-full rounded px-3 py-2 text-xs border border-[#4a3018] text-[var(--text-1)] hover:bg-[#1e1208] transition-colors"
                      >
                        <BookOpen size={11} /> Open Scene Builder
                      </button>
                    </div>
                  </div>
                )
              ) : (
                entries.map((entry) => {
                  const isSelected = selected?.id === entry.id;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      onClick={() => handleEntryClick(entry)}
                      className={[
                        "w-full text-left rounded px-3 py-2.5 border transition-all",
                        isSelected
                          ? "prep-entry-selected"
                          : "bg-[#1a1008]/90 border-[#5c3e23] hover:border-[#8a6236] hover:bg-[#1e1409]",
                      ].join(" ")}
                    >
                      <div className="font-heading text-[13px] text-[var(--text-1)] truncate">
                        {entry.title}
                      </div>
                      {entry.subtitle && (
                        <div className="text-[11px] text-[var(--gold)] truncate mt-0.5">
                          {entry.subtitle}
                        </div>
                      )}
                      {entry.description && (
                        <div className="text-[11px] text-[var(--text-2)] line-clamp-2 mt-0.5 leading-snug">
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
        className="flex-shrink-0 flex flex-col min-h-0 overflow-hidden detail-workspace"
        style={{ width: "320px" }}
      >
        <div className="flex-shrink-0 px-4 py-2.5 border-b border-[#2a1a0a] bg-[#110b06]">
          <div className="font-heading text-xs text-[var(--text-2)] uppercase tracking-[0.15em]">
            {selected ? selected.title : "Detail"}
          </div>
        </div>
        {selected ? (
          <DetailPanel
            entry={selected}
            onNavigate={onNavigate}
          />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center px-6 py-10 gap-5 text-center">
            <div className="w-12 h-12 rounded-full border border-[#3a2510] bg-[#130c06] flex items-center justify-center">
              <BookOpen size={20} className="text-[var(--text-2)] opacity-50" />
            </div>
            <div>
              <p className="font-heading text-sm text-[var(--text-1)] tracking-wide">Entry Details</p>
              <p className="text-xs text-[var(--text-2)] mt-1.5 leading-relaxed">
                Select an entry from the list to view its description, relationships, and available actions.
              </p>
            </div>
            {/* Ghosted action hints */}
            <div className="w-full pt-4 border-t border-[#2a1a0a] flex flex-col gap-2">
              <p className="text-[10px] uppercase tracking-[0.15em] text-[#4a3018] mb-1">Actions</p>
              {["Narrate", "Open in Live", "Assign Voice"].map((label) => (
                <button
                  key={label}
                  type="button"
                  disabled
                  className="w-full rounded px-3 py-1.5 text-xs border border-[#2a1a0a] text-[#3a2510] bg-transparent cursor-not-allowed font-heading tracking-wide"
                >
                  {label}
                </button>
              ))}
            </div>
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
