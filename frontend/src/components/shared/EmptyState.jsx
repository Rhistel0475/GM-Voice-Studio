/**
 * Placeholder for empty lists; optional icon and action. Fantasy panel styling.
 */
export default function EmptyState({ message = "Nothing here yet.", icon: Icon, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-[#5c3e23] bg-[#1a1008]/90 py-10 px-5 text-center shadow-inner">
      {Icon && (
        <Icon size={32} className="text-[var(--gold)]/70 shrink-0" aria-hidden />
      )}
      <p className="text-sm font-heading text-[var(--text-2)] tracking-wide">{message}</p>
      {action}
    </div>
  );
}
