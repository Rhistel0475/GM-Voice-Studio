/**
 * Controlled search input for the voice library. Matches Codex/NPC chat-input styling.
 */
export default function VoiceSearchInput({ value, onChange, placeholder = "Search voices…" }) {
  return (
    <input
      type="search"
      className="chat-input w-full"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label="Search voices"
    />
  );
}
