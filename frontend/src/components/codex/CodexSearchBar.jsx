/**
 * Search input for the Codex research view. Fantasy-styled.
 */
export default function CodexSearchBar({ value, onChange, placeholder = "Search codex…" }) {
  return (
    <input
      type="search"
      className="chat-input w-full"
      placeholder={placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      aria-label="Search codex"
    />
  );
}
