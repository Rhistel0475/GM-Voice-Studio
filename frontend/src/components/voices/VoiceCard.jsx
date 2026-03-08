import React from "react";
import { FantasyButton } from "../shared";
import { Play } from "lucide-react";

export default function VoiceCard({ voice, onPlaySample, onAssign }) {
  const name = voice?.name?.trim() || voice?.voice_id || "Unknown";
  return (
    <div className="portrait-card flex flex-col items-center rounded border border-[#6f4d2a] p-2">
      <div className="portrait-slot w-full" />
      <p className="mt-1 text-sm font-heading text-[var(--text-1)]">{name}</p>
      <FantasyButton
        variant="secondary"
        className="text-xs mt-1 w-full"
        onClick={() => onPlaySample(voice?.voice_id)}
      >
        <Play size={12} className="inline mr-1" /> Play sample
      </FantasyButton>
      {onAssign && (
        <FantasyButton variant="ghost" className="text-xs mt-1 w-full" onClick={() => onAssign(voice)}>
          Assign to NPC
        </FantasyButton>
      )}
    </div>
  );
}
