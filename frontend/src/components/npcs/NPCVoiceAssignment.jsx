import { FantasyButton } from "../shared";
import { Play, Wand2 } from "lucide-react";

export default function NPCVoiceAssignment({
  voices = [],
  selectedVoiceId,
  onVoiceChange,
  onPlaySample,
  playing,
  suggestion = null,
  onApplySuggestion,
}) {
  const suggestedVoiceId = suggestion?.voice_id || "";
  const suggestionApplied = !!suggestedVoiceId && suggestedVoiceId === selectedVoiceId;

  return (
    <div className="flex flex-col gap-2">
      {suggestedVoiceId && (
        <div className="rounded border border-[#7a5a32] bg-[#140d08] px-3 py-2">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="text-[10px] font-heading uppercase tracking-[0.2em] text-[var(--text-2)]">
                Suggested Voice
              </div>
              <div className="mt-1 flex items-center gap-2 text-sm text-[var(--gold)]">
                <Wand2 size={13} className="shrink-0" />
                <span className="truncate">{suggestion.voice_name || suggestedVoiceId}</span>
              </div>
              {(suggestion.tone || suggestion.style) && (
                <p className="mt-1 text-xs text-[var(--text-2)]">
                  {[suggestion.tone, suggestion.style].filter(Boolean).join(" · ")}
                </p>
              )}
            </div>
            {onApplySuggestion && (
              <FantasyButton
                variant="secondary"
                className="shrink-0"
                onClick={() => onApplySuggestion(suggestedVoiceId)}
                disabled={suggestionApplied}
              >
                {suggestionApplied ? "Suggested Voice Applied" : "Apply Suggested Voice"}
              </FantasyButton>
            )}
          </div>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <label className="field-wrap flex-1 min-w-[120px]">
          <span>Assign Voice</span>
          <select
            className="chat-input w-full"
            value={selectedVoiceId}
            onChange={(e) => onVoiceChange(e.target.value)}
          >
            <option value="">—</option>
            {voices.map((v) => (
              <option key={v.voice_id} value={v.voice_id}>
                {v.name?.trim() || v.voice_id}
              </option>
            ))}
          </select>
        </label>
        <FantasyButton variant="secondary" className="self-end" onClick={onPlaySample} disabled={playing}>
          {playing ? "Playing…" : <><Play size={14} className="inline mr-1" />Play sample</>}
        </FantasyButton>
      </div>
    </div>
  );
}
