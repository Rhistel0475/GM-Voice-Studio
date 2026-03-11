import { FantasyButton } from "../shared";
import { Play } from "lucide-react";

export default function NPCVoiceAssignment({
  voices = [],
  selectedVoiceId,
  onVoiceChange,
  onPlaySample,
  playing,
}) {
  return (
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
  );
}
