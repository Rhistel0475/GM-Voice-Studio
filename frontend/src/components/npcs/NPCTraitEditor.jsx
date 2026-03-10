import React, { useState } from "react";
import { Plus, X } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * Editable list of personality traits (chips); add/remove.
 */
export default function NPCTraitEditor({ value = [], onChange }) {
  const [input, setInput] = useState("");

  const add = () => {
    const t = input.trim();
    if (!t || value.includes(t)) return;
    onChange([...value, t]);
    setInput("");
  };

  const remove = (trait) => {
    onChange(value.filter((t) => t !== trait));
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Personality traits
      </div>
      <div className="flex flex-wrap gap-1">
        {value.map((trait) => (
          <span
            key={trait}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-[#5c3e23] bg-[#1a1008] text-sm text-[var(--ink-1)]"
          >
            {trait}
            <button
              type="button"
              className="p-0.5 hover:text-[var(--gold)] text-[var(--text-2)]"
              onClick={() => remove(trait)}
              aria-label={`Remove ${trait}`}
            >
              <X size={12} />
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          className="chat-input flex-1 text-sm"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
          placeholder="Add a trait…"
        />
        <FantasyButton variant="secondary" onClick={add} disabled={!input.trim()}>
          <Plus size={14} />
        </FantasyButton>
      </div>
    </div>
  );
}
