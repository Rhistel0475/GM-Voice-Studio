import React from "react";
import { ParchmentCard } from "../shared";
import { EmptyState } from "../shared";

export default function CodexDocumentViewer({ doc }) {
  if (!doc) {
    return <EmptyState message="Select a section and an item." />;
  }
  const title = doc.title ?? doc.name ?? "Untitled";
  const body = doc.body ?? doc.read_aloud ?? doc.personality ?? "";
  return (
    <div className="parchment flex-1 min-h-0 overflow-auto rounded">
      <h2 className="font-heading text-[var(--gold)] text-lg mb-2">{title}</h2>
      <div className="font-note text-[var(--ink-1)] whitespace-pre-wrap">{body}</div>
    </div>
  );
}
