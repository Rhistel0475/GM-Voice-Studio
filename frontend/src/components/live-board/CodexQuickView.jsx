import { FantasyButton, EmptyState } from "../shared";

function entryLabel(codexTab) {
  switch (codexTab) {
    case "npcs":
      return "No NPC knowledge loaded.";
    case "scenes":
      return "No scene knowledge loaded.";
    case "locations":
      return "No locations extracted yet.";
    case "quests":
      return "No quests linked to this campaign yet.";
    case "factions":
      return "No factions extracted yet.";
    default:
      return "No entries available.";
  }
}

export default function CodexQuickView({
  codexTab,
  entries = [],
  codexSelection,
  onSelect,
  onInsertIntoNarration,
  onSpeakNpc,
}) {
  const listItemBase =
    "border border-[#5c3e23] bg-[#1a1008]/90 rounded-md px-2.5 py-2 cursor-pointer transition-all hover:border-[var(--gold)] hover:bg-[#221808] hover:shadow-[0_0_8px_rgba(202,167,75,0.15)]";
  const listItemSelected = "ring-1 ring-[var(--gold)] border-[var(--gold)]";

  const renderList = () =>
    entries.length ? (
      entries.map((entry) => (
        <div
          key={entry.id}
          className={`${listItemBase} ${codexSelection?.id === entry.id ? listItemSelected : ""}`}
          onClick={() => onSelect(codexSelection?.id === entry.id ? null : entry)}
        >
          <div className="font-heading text-[var(--text-1)] text-sm">{entry.title}</div>
          {entry.subtitle && <div className="text-xs text-[var(--gold)] truncate">{entry.subtitle}</div>}
          {entry.description && <div className="text-xs text-[var(--text-2)] truncate">{entry.description}</div>}
        </div>
      ))
    ) : (
      <EmptyState message={entryLabel(codexTab)} />
    );

  const renderRelationshipChips = (relationships) => {
    if (!Array.isArray(relationships) || relationships.length === 0) return null;
    return (
      <div className="flex flex-wrap gap-1.5 mb-3">
        {relationships.slice(0, 8).map((relationship) => (
          <span
            key={relationship}
            className="rounded-full border border-[#6b4a28] bg-[#24170b] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-[var(--text-2)]"
          >
            {relationship}
          </span>
        ))}
      </div>
    );
  };

  const renderSummaryCard = () => {
    if (!codexSelection) return null;
    return (
      <div className="rounded-lg border border-[var(--gold)] bg-[#1a1008] p-3 shadow-[0_0_12px_rgba(202,167,75,0.12)]">
        <div className="font-heading text-[var(--gold)] text-sm mb-1">{codexSelection.title}</div>
        {codexSelection.subtitle && (
          <div className="text-xs text-[#d4b36a] mb-1">{codexSelection.subtitle}</div>
        )}
        <div className="text-xs text-[var(--text-2)] mb-3 whitespace-pre-wrap">
          {codexSelection.description || "No description available yet."}
        </div>
        {renderRelationshipChips(codexSelection.relationships)}
        <div className="flex flex-wrap gap-2">
          <FantasyButton
            variant="secondary"
            className="text-xs transition-all hover:brightness-110"
            onClick={() => onInsertIntoNarration?.(codexSelection.narrationText || codexSelection.title)}
          >
            Narrate
          </FantasyButton>
          <FantasyButton
            variant="ghost"
            className="text-xs transition-all hover:brightness-110"
            onClick={() => codexSelection.speakTargetName && onSpeakNpc?.(codexSelection.speakTargetName)}
            disabled={!codexSelection.speakTargetName || !onSpeakNpc}
          >
            Speak
          </FantasyButton>
        </div>
      </div>
    );
  };

  return (
    <div className="flex-1 min-h-0 overflow-auto flex flex-col gap-3">
      <div className="flex flex-col gap-2 min-h-0 overflow-auto">
        {renderList()}
      </div>
      {renderSummaryCard()}
    </div>
  );
}
