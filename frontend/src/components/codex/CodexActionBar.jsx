import React from "react";
import { FantasyButton } from "../shared";

/**
 * Actions: Summarize, Ask Question (input + button), and note about Extract NPCs/Locations.
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
}) {
  return (
    <div className="flex flex-wrap gap-2 border-t border-[#5c3e23] pt-2">
      <FantasyButton
        variant="secondary"
        onClick={onSummarize}
        disabled={summarizeDisabled || summarizeLoading}
      >
        {summarizeLoading ? "…" : "Summarize"}
      </FantasyButton>
      <span className="flex-1" />
      <input
        type="text"
        className="chat-input flex-1 min-w-[120px]"
        placeholder="Ask a question…"
        value={askQuery}
        onChange={(e) => onAskQueryChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onAskQuestion()}
      />
      <FantasyButton
        variant="primary"
        onClick={onAskQuestion}
        disabled={askDisabled || askLoading}
      >
        {askLoading ? "…" : "Ask Question"}
      </FantasyButton>
      <div className="w-full text-xs text-[var(--text-2)] mt-1">
        Extract NPCs / Extract Locations: run AI parse in Library to populate this Codex.
      </div>
    </div>
  );
}
