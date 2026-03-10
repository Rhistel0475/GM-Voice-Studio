/**
 * Controlled search input; matches CodexSearchBar styling (chat-input).
 */
export default function NPCSearchInput({ value, onChange, placeholder = "Search NPCs…" }) {
  return (
    <input
      type="search"
      className="chat-input w-full"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label="Search NPCs"
    />
  );
}
