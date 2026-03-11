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
  return (
    <div className="flex flex-col gap-2">
      {suggestion?.presetName && (
        <div className="npc-voice-suggestion">
          <Wand2 size={12} className="inline-block mr-1 text-[var(--gold)]" />
          <span className="npc-voice-suggestion-label">
            Suggested: <strong>{suggestion.presetName}</strong>
          </span>
          <span className="npc-voice-suggestion-reason">{suggestion.reason}</span>
          {suggestion.candidateVoices.length > 0 && onApplySuggestion && (
            <button
              type="button"
              className="npc-voice-suggestion-apply"
              onClick={() => onApplySuggestion(suggestion.candidateVoices[0].voice_id || suggestion.candidateVoices[0].id)}
              title={`Apply: ${suggestion.candidateVoices[0].name}`}
            >
              Apply
            </button>
          )}
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
