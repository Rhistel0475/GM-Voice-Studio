import { FantasyButton } from "../shared";

export default function NarrationComposer({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder = "Ask Co-DM...",
}) {
  return (
    <div className="flex gap-2">
      <input
        type="text"
        placeholder={placeholder}
        className="chat-input flex-1"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSubmit()}
      />
      <FantasyButton
        type="button"
        variant="primary"
        onClick={onSubmit}
        disabled={disabled || !(value || "").trim()}
      >
        {disabled ? "..." : "Send"}
      </FantasyButton>
    </div>
  );
}
