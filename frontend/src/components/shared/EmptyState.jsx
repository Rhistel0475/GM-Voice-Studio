import React from "react";

/**
 * Placeholder for empty lists; optional icon and action.
 */
export default function EmptyState({ message = "Nothing here yet.", icon: Icon, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded border border-[#5a3e1b] bg-[#1a0f06]/50 py-8 px-4 text-center text-[#9c7a3a]">
      {Icon && <Icon size={32} className="opacity-70" />}
      <p className="text-sm font-heading">{message}</p>
      {action}
    </div>
  );
}
