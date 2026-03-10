import React from "react";

/**
 * Shared search input for Codex, NPC list, Voice library. Uses chat-input styling.
 */
export default function SearchInput({
  value,
  onChange,
  placeholder = "Search…",
  ariaLabel = "Search",
  className = "",
}) {
  return (
    <input
      type="search"
      className={`chat-input w-full ${className}`.trim()}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={ariaLabel}
    />
  );
}
