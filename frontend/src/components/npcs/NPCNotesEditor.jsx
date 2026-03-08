import React from "react";

/**
 * Textarea for GM notes on the NPC.
 */
export default function NPCNotesEditor({ value = "", onChange, placeholder = "GM notes…" }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Notes
      </div>
      <textarea
        className="chat-input w-full min-h-[80px] resize-y text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={3}
      />
    </div>
  );
}
