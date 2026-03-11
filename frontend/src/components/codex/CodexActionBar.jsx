import { FantasyButton } from "../shared";
import { FileText, MessageCircle, MapPin, Users, LayoutGrid } from "lucide-react";

/**
 * Actions: Summarize, Extract, Add to Live Board; Ask Question. Research-desk style.
 */
export default function CodexActionBar({
  onSummarize,
  summarizeDisabled,
  summarizeLoading,
  askQuery,
  onAskQueryChange,
  onAskQuestion,
  askDisabled,
  askLoading,
  onExtractNpcs,
  onExtractLocations,
  onAddToLiveBoard,
}) {
  return (
    <div className="flex flex-col gap-3 shrink-0 border-t border-[#5c3e23] pt-3">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Actions
      </div>
      <div className="flex flex-wrap gap-2">
        <FantasyButton
          variant="secondary"
          onClick={onSummarize}
          disabled={summarizeDisabled || summarizeLoading}
        >
          {summarizeLoading ? "…" : <><FileText size={14} className="inline mr-1.5 shrink-0" />Summarize</>}
        </FantasyButton>
        <FantasyButton
          variant="secondary"
          onClick={onExtractNpcs || (() => {})}
          title="Extract NPCs from content"
        >
          <Users size={14} className="inline mr-1.5 shrink-0" />
          Extract NPCs
        </FantasyButton>
        <FantasyButton
          variant="secondary"
          onClick={onExtractLocations || (() => {})}
          title="Extract locations"
        >
          <MapPin size={14} className="inline mr-1.5 shrink-0" />
          Extract Locations
        </FantasyButton>
        <FantasyButton
          variant="secondary"
          onClick={onAddToLiveBoard || (() => {})}
          title="Add to Live Board"
        >
          <LayoutGrid size={14} className="inline mr-1.5 shrink-0" />
          Add to Live Board
        </FantasyButton>
      </div>
      <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-2">
        <label className="flex-1 min-w-0 flex flex-col gap-1">
          <span className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">Ask</span>
          <input
            type="text"
            className="chat-input w-full min-w-0"
            placeholder="Ask a question about this entry…"
            value={askQuery}
            onChange={(e) => onAskQueryChange(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onAskQuestion()}
          />
        </label>
        <FantasyButton
          variant="primary"
          onClick={onAskQuestion}
          disabled={askDisabled || askLoading}
          className="sm:self-end"
        >
          {askLoading ? "…" : <><MessageCircle size={14} className="inline mr-1.5 shrink-0" />Ask Question</>}
        </FantasyButton>
      </div>
    </div>
  );
}
