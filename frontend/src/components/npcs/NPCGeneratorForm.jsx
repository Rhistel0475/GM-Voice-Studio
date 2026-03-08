import React from "react";
import { FantasyButton } from "../shared";
import { Play } from "lucide-react";

export default function NPCGeneratorForm({
  genre,
  onGenreChange,
  location,
  onLocationChange,
  npcName,
  onNpcNameChange,
  role,
  onRoleChange,
  onGenerate,
  onRegenerate,
  generating,
  error,
  personalityText,
  voiceSelect,
  onPlaySample,
  playing,
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2">
        <div className="field-wrap">
          <span>Genre</span>
          <input
            className="chat-input"
            value={genre}
            onChange={(e) => onGenreChange(e.target.value)}
            placeholder="e.g. 1930s noir fantasy"
          />
        </div>
        <div className="field-wrap">
          <span>Location</span>
          <input
            className="chat-input"
            value={location}
            onChange={(e) => onLocationChange(e.target.value)}
            placeholder="e.g. The Silver Dagger Inn"
          />
        </div>
        <div className="field-wrap">
          <span>Name / Type</span>
          <input
            className="chat-input"
            value={npcName}
            onChange={(e) => onNpcNameChange(e.target.value)}
            placeholder="Viktor Crane"
          />
        </div>
        <div className="field-wrap">
          <span>Role</span>
          <input
            className="chat-input"
            value={role}
            onChange={(e) => onRoleChange(e.target.value)}
            placeholder="corrupt detective"
          />
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <FantasyButton
          variant="secondary"
          onClick={onGenerate}
          disabled={!npcName?.trim() || generating}
        >
          {generating ? "Generating…" : "Generate NPC"}
        </FantasyButton>
        <FantasyButton
          variant="secondary"
          onClick={onRegenerate}
          disabled={!npcName?.trim() || generating}
          title="Regenerate personality"
        >
          Regenerate Personality
        </FantasyButton>
      </div>
      {error && <div className="text-xs text-red-400">{error}</div>}
      {personalityText && (
        <>
          <div className="border border-[#4f341f] bg-[#0e0906] rounded p-2 max-h-48 overflow-y-auto">
            <pre className="whitespace-pre-wrap font-mono text-xs text-[#e6c785]">{personalityText}</pre>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {voiceSelect}
            <FantasyButton variant="secondary" onClick={onPlaySample} disabled={playing}>
              {playing ? "Playing…" : <><Play size={14} className="inline mr-1" />Play sample</>}
            </FantasyButton>
          </div>
        </>
      )}
    </div>
  );
}
