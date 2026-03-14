import { useState } from "react";
import { Plus, X } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * List of secrets (strings); add/remove.
 */
export default function NPCSecretsPanel({ value = [], onChange }) {
  const [input, setInput] = useState("");

  const add = () => {
    const s = input.trim();
    if (!s) return;
    onChange([...value, s]);
    setInput("");
  };

  const remove = (secret) => {
    onChange(value.filter((s) => s !== secret));
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Secrets
      </div>
      <ul className="space-y-1 text-sm text-[var(--ink-1)]">
        {value.map((secret) => (
          <li key={secret} className="flex items-start gap-2 border border-[#5c3e23] bg-[#0e0906] rounded p-2">
            <span className="flex-1 min-w-0">{secret}</span>
            <button
              type="button"
              className="shrink-0 p-0.5 hover:text-[var(--gold)] text-[var(--text-2)]"
              onClick={() => remove(secret)}
              aria-label="Remove secret"
            >
              <X size={12} />
            </button>
          </li>
        ))}
      </ul>
      <div className="flex gap-2">
        <input
          type="text"
          className="chat-input flex-1 text-sm"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
          placeholder="Add a secret…"
        />
        <FantasyButton variant="secondary" onClick={add} disabled={!input.trim()}>
          <Plus size={14} />
        </FantasyButton>
      </div>
    </div>
  );
}
