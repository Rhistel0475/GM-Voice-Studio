import React from "react";
import { FantasyButton, EmptyState } from "../shared";

/**
 * Renders the list of entries for the selected codex tab. Selecting an entry shows a quick summary card.
 */
export default function CodexQuickView({
  codexTab,
  documents = [],
  npcs = [],
  locations = [],
  codexSelection,
  onSelect,
  onInsertIntoNarration,
}) {
  const listItemBase =
    "border border-[#5c3e23] bg-[#1a1008]/90 rounded-md px-2.5 py-2 cursor-pointer transition-all hover:border-[var(--gold)] hover:bg-[#221808] hover:shadow-[0_0_8px_rgba(202,167,75,0.15)]";
  const listItemSelected = "ring-1 ring-[var(--gold)] border-[var(--gold)]";

  const renderDocuments = () =>
    documents.map((doc) => (
      <div
        key={doc.id}
        className={`${listItemBase} ${codexSelection?.id === doc.id ? listItemSelected : ""}`}
        onClick={() => onSelect(codexSelection?.id === doc.id ? null : doc)}
      >
        <div className="font-heading text-[var(--text-1)] text-sm">{doc.title}</div>
        <div className="text-xs text-[var(--text-2)] truncate">{doc.summary}</div>
      </div>
    ));

  const renderNpcs = () =>
    npcs.length ? (
      npcs.map((n) => (
        <div
          key={n.name}
          className={`${listItemBase} ${codexSelection?.name === n.name ? listItemSelected : ""}`}
          onClick={() => onSelect(codexSelection?.name === n.name ? null : n)}
        >
          <div className="font-heading text-[var(--text-1)] text-sm">{n.name}</div>
          <div className="text-xs text-[var(--text-2)] truncate">{n.role || ""}</div>
        </div>
      ))
    ) : (
      <EmptyState message="No NPCs loaded." />
    );

  const renderLocations = () =>
    locations.length ? (
      locations.map((loc) => {
        const key = typeof loc === "string" ? loc : loc.name || loc;
        const name = typeof loc === "string" ? loc : loc.name || loc;
        return (
          <div
            key={key}
            className={`${listItemBase} ${codexSelection === loc ? listItemSelected : ""}`}
            onClick={() => onSelect(codexSelection === loc ? null : loc)}
          >
            <div className="font-heading text-[var(--text-1)] text-sm">{name}</div>
          </div>
        );
      })
    ) : (
      <EmptyState message="No locations extracted. Use Library to parse adventures." />
    );

  const renderRules = () => (
    <EmptyState message="Rules lookup (RAG) — ask in Live Session or use Co-DM query." />
  );

  const renderSummaryCard = () => {
    if (!codexSelection || !onInsertIntoNarration) return null;
    if (codexTab === "documents" && codexSelection.id) {
      return (
        <div className="rounded-lg border border-[var(--gold)] bg-[#1a1008] p-3 shadow-[0_0_12px_rgba(202,167,75,0.12)]">
          <div className="font-heading text-[var(--gold)] text-sm mb-1">Summary</div>
          <div className="text-xs text-[var(--text-2)] mb-2">{codexSelection.summary}</div>
          <FantasyButton variant="secondary" className="text-xs transition-all hover:brightness-110" onClick={() => onInsertIntoNarration(codexSelection.title)}>
            Insert into narration
          </FantasyButton>
        </div>
      );
    }
    if (codexTab === "npcs" && codexSelection.name) {
      return (
        <div className="rounded-lg border border-[var(--gold)] bg-[#1a1008] p-3 shadow-[0_0_12px_rgba(202,167,75,0.12)]">
          <div className="font-heading text-[var(--gold)] text-sm mb-1">{codexSelection.name}</div>
          {codexSelection.role && <div className="text-xs text-[var(--text-2)] mb-1">{codexSelection.role}</div>}
          {codexSelection.personality && <div className="text-xs text-[var(--text-2)] mb-2 line-clamp-3">{codexSelection.personality}</div>}
          <FantasyButton variant="secondary" className="text-xs transition-all hover:brightness-110" onClick={() => onInsertIntoNarration(codexSelection.personality || codexSelection.name)}>
            Insert into narration
          </FantasyButton>
        </div>
      );
    }
    if (codexTab === "locations") {
      const name = typeof codexSelection === "string" ? codexSelection : codexSelection.name || codexSelection;
      return (
        <div className="rounded-lg border border-[var(--gold)] bg-[#1a1008] p-3 shadow-[0_0_12px_rgba(202,167,75,0.12)]">
          <div className="font-heading text-[var(--gold)] text-sm mb-1">Location</div>
          <div className="text-xs text-[var(--text-2)] mb-2">{name}</div>
          <FantasyButton variant="secondary" className="text-xs transition-all hover:brightness-110" onClick={() => onInsertIntoNarration(name)}>
            Insert into narration
          </FantasyButton>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex-1 min-h-0 overflow-auto flex flex-col gap-3">
      <div className="flex flex-col gap-2 min-h-0 overflow-auto">
        {codexTab === "documents" && renderDocuments()}
        {codexTab === "npcs" && renderNpcs()}
        {codexTab === "locations" && renderLocations()}
        {codexTab === "rules" && renderRules()}
      </div>
      {renderSummaryCard()}
    </div>
  );
}
