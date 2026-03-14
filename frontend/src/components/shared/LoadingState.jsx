/**
 * Loading placeholder (spinner or message). Fantasy panel styling.
 */
export default function LoadingState({ message = "Loading…" }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10 px-5 rounded-lg border border-[#5c3e23] bg-[#1a1008]/70">
      <div
        className="h-8 w-8 animate-spin rounded-full border-2 border-[#5c3e23] border-t-[var(--gold)]"
        aria-hidden
      />
      <p className="text-sm font-heading text-[var(--text-2)] tracking-wide">{message}</p>
    </div>
  );
}
