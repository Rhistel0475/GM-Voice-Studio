import { ParchmentCard } from "../shared";

/**
 * Card for displaying a knowledge snippet (e.g. summarize result, ask result). Parchment style.
 */
export default function KnowledgeCard({ title, children, className = "" }) {
  return (
    <ParchmentCard title={title} className={`border border-[#a17a42] ${className}`.trim()}>
      <div className="text-sm text-[var(--ink-1)] leading-relaxed">{children}</div>
    </ParchmentCard>
  );
}
