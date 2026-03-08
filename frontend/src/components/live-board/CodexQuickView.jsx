import React from "react";
import { FantasyButton, EmptyState } from "../shared";

/**
 * Renders the list of items for the selected codex tab and "Insert into narration" for the selection.
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
  const renderDocuments = () =>
    documents.map((doc) => (
      <div
        key={doc.id}
        className={`border border-[#5c3e23] bg-[#1a1008] p-2 cursor-pointer hover:border-[var(--gold)] ${codexSelection?.id === doc.id ? "ring-1 ring-[var(--gold)]" : ""}`}
        onClick={() => onSelect(codexSelection?.id === doc.id ? null : doc)}
      >
        <div className="font-heading text-[var(--text-1)] text-sm">{doc.title}</div>
        <div className="text-xs text-[var(--text-2)]">{doc.summary}</div>
        {codexSelection?.id === doc.id && onInsertIntoNarration && (
          <FantasyButton
            variant="secondary"
            className="mt-2 text-xs"
            onClick={(e) => {
              e.stopPropagation();
              onInsertIntoNarration(doc.title);
            }}
          >
            Insert into narration
          </FantasyButton>
        )}
      </div>
    ));

  const renderNpcs = () =>
    npcs.length ? (
      npcs.map((n) => (
        <div
          key={n.name}
          className={`border border-[#5c3e23] bg-[#1a1008] p-2 cursor-pointer hover:border-[var(--gold)] ${codexSelection?.name === n.name ? "ring-1 ring-[var(--gold)]" : ""}`}
          onClick={() => onSelect(codexSelection?.name === n.name ? null : n)}
        >
          <div className="font-heading text-[var(--text-1)] text-sm">{n.name}</div>
          <div className="text-xs text-[var(--text-2)]">{n.role || ""}</div>
          {n.personality && <div className="text-xs text-[var(--text-2)] mt-1 line-clamp-2">{n.personality}</div>}
          {codexSelection?.name === n.name && onInsertIntoNarration && (
            <FantasyButton
              variant="secondary"
              className="mt-2 text-xs"
              onClick={(e) => {
                e.stopPropagation();
                onInsertIntoNarration(n.personality || n.name);
              }}
            >
              Insert into narration
            </FantasyButton>
          )}
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
            className={`border border-[#5c3e23] bg-[#1a1008] p-2 cursor-pointer hover:border-[var(--gold)] ${codexSelection === loc ? "ring-1 ring-[var(--gold)]" : ""}`}
            onClick={() => onSelect(codexSelection === loc ? null : loc)}
          >
            <div className="font-heading text-[var(--text-1)] text-sm">{name}</div>
            {codexSelection === loc && onInsertIntoNarration && (
              <FantasyButton
                variant="secondary"
                className="mt-2 text-xs"
                onClick={(e) => {
                  e.stopPropagation();
                  onInsertIntoNarration(name);
                }}
              >
                Insert into narration
              </FantasyButton>
            )}
          </div>
        );
      })
    ) : (
      <EmptyState message="No locations extracted. Use Library to parse adventures." />
    );

  const renderRules = () => (
    <EmptyState message="Rules lookup (RAG) — ask in Live Session or use Co-DM query." />
  );

  return (
    <div className="flex-1 min-h-0 overflow-auto flex flex-col gap-2">
      {codexTab === "documents" && renderDocuments()}
      {codexTab === "npcs" && renderNpcs()}
      {codexTab === "locations" && renderLocations()}
      {codexTab === "rules" && renderRules()}
    </div>
  );
}
