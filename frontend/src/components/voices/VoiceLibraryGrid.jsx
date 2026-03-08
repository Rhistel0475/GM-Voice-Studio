import React from "react";
import VoiceCard from "./VoiceCard";
import VoiceListItem from "./VoiceListItem";
import { EmptyState } from "../shared";

/**
 * Voices as list or grid. viewMode "list" | "grid". Uses voice.id or voice.voice_id as key.
 */
export default function VoiceLibraryGrid({
  voices = [],
  selectedVoiceId,
  onPlaySample,
  onAssign,
  onSelectVoice,
  viewMode = "grid",
}) {
  if (!voices.length) {
    return <EmptyState message="No voices. Clone one or load from the server." />;
  }
  if (viewMode === "list") {
    return (
      <div className="space-y-1.5 overflow-y-auto flex-1 min-h-0">
        {voices.map((v) => (
          <VoiceListItem
            key={v.voice_id || v.id}
            voice={v}
            selected={selectedVoiceId === (v.voice_id || v.id)}
            onSelect={onSelectVoice}
            onPlaySample={onPlaySample}
          />
        ))}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 overflow-auto">
      {voices.map((v) => (
        <VoiceCard
          key={v.voice_id || v.id}
          voice={v}
          selected={selectedVoiceId === (v.voice_id || v.id)}
          onPlaySample={onPlaySample}
          onAssign={onAssign}
          onSelect={onSelectVoice}
        />
      ))}
    </div>
  );
}
