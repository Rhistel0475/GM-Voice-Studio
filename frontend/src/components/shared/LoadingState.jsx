import React from "react";

/**
 * Loading placeholder (spinner or message).
 */
export default function LoadingState({ message = "Loading…" }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-8 px-4 text-[#9c7a3a]">
      <div
        className="h-8 w-8 animate-spin rounded-full border-2 border-[#5a3e1b] border-t-[var(--gold)]"
        aria-hidden
      />
      <p className="text-sm font-heading">{message}</p>
    </div>
  );
}
