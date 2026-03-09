import React, { useState } from "react";
import { ParchmentCard, EmptyState } from "../shared";
import CodexActionBar from "./CodexActionBar";
import KnowledgeCard from "./KnowledgeCard";

const CATEGORY_LABELS = {
  document: "Document",
  npc: "NPC",
  location: "Location",
  rule: "Rule",
  lore: "Lore",
  faction: "Faction",
};

/**
 * Right column: selected item detail, metadata, summary/content, and action bar.
 * Manages Summarize / Ask Question state and calls authFetch("/brain/query").
 */
export default function CodexDetailPanel({
  item,
  authFetch,
  onSummarizeResult,
  onAskResult,
  onAddToLiveBoard,
}) {
  const [askQuery, setAskQuery] = useState("");
  const [askResult, setAskResult] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [summarizeResult, setSummarizeResult] = useState("");
  const [isSummarizing, setIsSummarizing] = useState(false);

  const handleSummarize = async () => {
    if (!item || isSummarizing) return;
    setIsSummarizing(true);
    setSummarizeResult("");
    try {
      const text = item.content || item.summary || item.excerpt || item.title || "";
      const res = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `Summarize the following for the GM:\n\n${text.slice(0, 4000)}`,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const result = data?.answer || data?.response || JSON.stringify(data);
      setSummarizeResult(result);
      onSummarizeResult?.(result);
    } catch (e) {
      setSummarizeResult(e?.message || "Summarize failed.");
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!askQuery.trim() || isAsking) return;
    setIsAsking(true);
    setAskResult("");
    try {
      const res = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: askQuery.trim() }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const result = data?.answer || data?.response || JSON.stringify(data);
      setAskResult(result);
      onAskResult?.(result);
    } catch (e) {
      setAskResult(e?.message || "Failed to get answer.");
    } finally {
      setIsAsking(false);
    }
  };

  const stubExtractNpcs = () => {
    if (typeof window !== "undefined") window.alert("Extract NPCs — coming soon.");
  };
  const stubExtractLocations = () => {
    if (typeof window !== "undefined") window.alert("Extract Locations — coming soon.");
  };
  const stubAddToLiveBoard = () => {
    if (typeof window !== "undefined") window.alert("Add to Live Board — coming soon.");
  };
  const handleAddToLiveBoard = onAddToLiveBoard ? () => onAddToLiveBoard(item) : stubAddToLiveBoard;

  if (!item) {
    return (
      <div className="flex flex-col min-h-0 flex-1 items-center justify-center p-4">
        <div className="w-full max-w-sm">
          <EmptyState message="Select an item from the list to view details and use actions." />
        </div>
      </div>
    );
  }

  const categoryLabel = CATEGORY_LABELS[item.category] || item.category;
  const body = item.content || item.excerpt || item.summary || "";

  return (
    <div className="flex flex-col min-h-0 flex-1 gap-3">
      <div className="flex flex-col gap-1.5 flex-shrink-0">
        <h2 className="font-heading text-[var(--gold)] text-lg">{item.title}</h2>
        <div className="h-px w-12 bg-[var(--gold)]/50" aria-hidden />
        <div className="flex flex-wrap gap-2 text-xs text-[var(--text-2)]">
          <span className="uppercase tracking-wide text-[var(--candle-glow)]/90">{categoryLabel}</span>
          {item.campaign && <span>{item.campaign}</span>}
          {item.updatedAt && <span>{item.updatedAt}</span>}
          {item.source && <span>{item.source}</span>}
        </div>
        {item.tags?.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {item.tags.map((t) => (
              <span
                key={t}
                className="text-[10px] px-1.5 py-0.5 rounded border border-[#5c3e23] text-[var(--text-2)] bg-[#1a1008]"
              >
                {t}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="parchment rounded border border-[#a17a42] flex-1 min-h-0 overflow-auto p-3 text-[var(--ink-1)] text-sm whitespace-pre-wrap leading-relaxed">
        {body}
      </div>
      <CodexActionBar
        onSummarize={handleSummarize}
        summarizeDisabled={!item}
        summarizeLoading={isSummarizing}
        askQuery={askQuery}
        onAskQueryChange={setAskQuery}
        onAskQuestion={handleAskQuestion}
        askDisabled={!askQuery.trim()}
        askLoading={isAsking}
        onExtractNpcs={stubExtractNpcs}
        onExtractLocations={stubExtractLocations}
        onAddToLiveBoard={handleAddToLiveBoard}
      />
      {summarizeResult && (
        <KnowledgeCard title="Summary">{summarizeResult}</KnowledgeCard>
      )}
      {askResult && (
        <KnowledgeCard title="Answer" className="border-t border-[#a17a42] mt-2">
          {askResult}
        </KnowledgeCard>
      )}
    </div>
  );
}
