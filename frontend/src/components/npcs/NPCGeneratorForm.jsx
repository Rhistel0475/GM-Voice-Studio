import { useState } from "react";
import { FantasyButton } from "../shared";
import { Play, Plus, X } from "lucide-react";
import NPCTraitEditor from "./NPCTraitEditor";
import NPCSecretsPanel from "./NPCSecretsPanel";
import NPCNotesEditor from "./NPCNotesEditor";
import NPCVoicePreferenceField from "./NPCVoicePreferenceField";

/**
 * Inline list editor for goals/quirks; fantasy labels and chat-input.
 */
function GoalsQuirksList({ label, value = [], onChange, placeholder }) {
  const [input, setInput] = useState("");
  const add = () => {
    const t = input.trim();
    if (!t) return;
    onChange([...value, t]);
    setInput("");
  };
  const remove = (item) => onChange(value.filter((x) => x !== item));
  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        {label}
      </div>
      <ul className="space-y-1">
        {value.map((item) => (
          <li key={item} className="flex items-center gap-2 text-sm text-[var(--ink-1)] border border-[#5c3e23] bg-[#0e0906] rounded px-2 py-1">
            <span className="flex-1 truncate">{item}</span>
            <button type="button" className="p-0.5 hover:text-[var(--gold)] text-[var(--text-2)]" onClick={() => remove(item)} aria-label="Remove">
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
          placeholder={placeholder}
        />
        <FantasyButton variant="secondary" onClick={add} disabled={!input.trim()}>
          <Plus size={14} />
        </FantasyButton>
      </div>
    </div>
  );
}

export default function NPCGeneratorForm({
  draft,
  onDraftChange,
  genre,
  onGenreChange,
  npcName,
  onNpcNameChange,
  onGenerate,
  onRegenerate,
  onRegenerateBackstory,
  generating,
  error,
  personalityText,
  voices = [],
  selectedVoiceId,
  onVoiceChange,
  onPlaySample,
  playing,
}) {
  const setDraft = (patch) => onDraftChange({ ...draft, ...patch });
  const d = draft || {};

  return (
    <div className="flex flex-col gap-4 overflow-auto">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Summon
      </div>
      <div className="grid grid-cols-2 gap-2">
        <label className="field-wrap">
          <span>Genre</span>
          <input
            className="chat-input"
            value={genre}
            onChange={(e) => onGenreChange(e.target.value)}
            placeholder="e.g. 1930s noir fantasy"
          />
        </label>
        <label className="field-wrap">
          <span>Name / Type</span>
          <input
            className="chat-input"
            value={npcName}
            onChange={(e) => onNpcNameChange(e.target.value)}
            placeholder="Viktor Crane"
          />
        </label>
        <label className="field-wrap">
          <span>Role / Archetype</span>
          <input
            className="chat-input"
            value={d.role ?? ""}
            onChange={(e) => setDraft({ role: e.target.value })}
            placeholder="e.g. suspicious apothecary"
          />
        </label>
        <label className="field-wrap">
          <span>Profession</span>
          <input
            className="chat-input"
            value={d.profession ?? ""}
            onChange={(e) => setDraft({ profession: e.target.value })}
            placeholder="e.g. apothecary"
          />
        </label>
        <label className="field-wrap">
          <span>Faction</span>
          <input
            className="chat-input"
            value={d.faction ?? ""}
            onChange={(e) => setDraft({ faction: e.target.value })}
            placeholder="e.g. Independent"
          />
        </label>
        <label className="field-wrap">
          <span>Location</span>
          <input
            className="chat-input"
            value={d.location ?? ""}
            onChange={(e) => setDraft({ location: e.target.value })}
            placeholder="e.g. Docks district"
          />
        </label>
      </div>

      <div className="border-t border-[#5c3e23] pt-3 mt-1">
        <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2">
          Character
        </div>
        <NPCTraitEditor
        value={d.personalityTraits ?? []}
        onChange={(personalityTraits) => setDraft({ personalityTraits })}
      />
      <GoalsQuirksList
        label="Goals"
        value={d.goals ?? []}
        onChange={(goals) => setDraft({ goals })}
        placeholder="Add a goal…"
      />
      <NPCSecretsPanel
        value={d.secrets ?? []}
        onChange={(secrets) => setDraft({ secrets })}
      />
      <GoalsQuirksList
        label="Quirks"
        value={d.quirks ?? []}
        onChange={(quirks) => setDraft({ quirks })}
        placeholder="Add a quirk…"
      />
      <NPCNotesEditor
        value={d.notes ?? ""}
        onChange={(notes) => setDraft({ notes })}
      />
      <NPCVoicePreferenceField
        voices={voices}
        selectedVoiceId={selectedVoiceId}
        onVoiceChange={onVoiceChange}
      />
      </div>
      <div className="border-t border-[#5c3e23] pt-3">
        <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2">
          Generate
        </div>
      <div className="flex flex-wrap gap-2 shrink-0">
        <FantasyButton
          variant="primary"
          onClick={onGenerate}
          disabled={!npcName?.trim() || generating}
        >
          {generating ? "Generating…" : "Generate NPC"}
        </FantasyButton>
        <FantasyButton variant="secondary" onClick={onRegenerate} disabled={!npcName?.trim() || generating}>
          Regenerate Personality
        </FantasyButton>
          <FantasyButton variant="secondary" onClick={onRegenerateBackstory}>
          Regenerate Backstory
        </FantasyButton>
      </div>
      {error && <div className="text-xs text-red-400">{error}</div>}
      {personalityText && (
        <>
          <div className="parchment rounded border border-[#a17a42] p-3 max-h-40 overflow-y-auto">
            <pre className="whitespace-pre-wrap font-mono text-xs text-[var(--ink-1)]">{personalityText}</pre>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <FantasyButton variant="secondary" onClick={onPlaySample} disabled={playing}>
              {playing ? "Playing…" : <><Play size={14} className="inline mr-1" />Play sample</>}
            </FantasyButton>
          </div>
        </>
      )}
      </div>
    </div>
  );
}
