import React from "react";
import { ParchmentCard } from "../shared";

/**
 * Card for displaying a knowledge snippet (e.g. summarize result, ask result).
 */
export default function KnowledgeCard({ title, children, className = "" }) {
  return (
    <ParchmentCard title={title} className={className}>
      <div className="text-sm text-[var(--ink-1)]">{children}</div>
    </ParchmentCard>
  );
}
