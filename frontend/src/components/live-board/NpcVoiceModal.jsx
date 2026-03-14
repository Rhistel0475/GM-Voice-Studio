import { FantasyButton, ModalShell } from "../shared";

export default function NpcVoiceModal({
  open,
  mode = "speak",
  npc,
  value,
  onChange,
  onClose,
  onSubmit,
  busy = false,
  error = "",
  generatedText = "",
}) {
  if (!open || !npc) return null;

  const isGenerate = mode === "generate";
  const title = isGenerate ? `AI Response: ${npc.name || "NPC"}` : `Speak as ${npc.name || "NPC"}`;
  const promptLabel = isGenerate ? "Player input" : "Dialogue";
  const promptPlaceholder = isGenerate
    ? "What did the players say or do?"
    : "What should this NPC say?";

  return (
    <ModalShell title={title} onClose={busy ? undefined : onClose}>
      <div className="flex flex-col gap-3">
        <p className="text-sm text-[var(--text-2)]">
          {isGenerate
            ? "Generate a short in-character response and speak it immediately."
            : "Send a direct line to the NPC's assigned voice."}
        </p>

        <label className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-[#9b7440]">{promptLabel}</span>
          <textarea
            className="chat-input min-h-[112px] resize-y"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={promptPlaceholder}
            disabled={busy}
          />
        </label>

        {error ? <div className="text-xs text-red-400">{error}</div> : null}

        {generatedText ? (
          <div className="rounded border border-[#5c3e23] bg-[#1a1008]/70 p-3">
            <div className="text-[10px] uppercase tracking-wide text-[#9b7440] mb-1">
              Generated dialogue
            </div>
            <div className="text-sm text-[var(--text-1)] whitespace-pre-wrap">{generatedText}</div>
          </div>
        ) : null}

        <div className="flex flex-wrap justify-end gap-2">
          <FantasyButton variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </FantasyButton>
          <FantasyButton
            variant={isGenerate ? "primary" : "secondary"}
            onClick={onSubmit}
            disabled={busy || !(value || "").trim()}
          >
            {busy ? "Working..." : isGenerate ? "Generate & Speak" : "Speak"}
          </FantasyButton>
        </div>
      </div>
    </ModalShell>
  );
}
