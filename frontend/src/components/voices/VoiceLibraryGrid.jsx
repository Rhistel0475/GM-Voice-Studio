import React from "react";
import VoiceCard from "./VoiceCard";
import { EmptyState } from "../shared";

export default function VoiceLibraryGrid({ voices = [], onPlaySample, onAssign }) {
  if (!voices.length) {
    return <EmptyState message="No voices. Clone one below." />;
  }
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 overflow-auto">
      {voices.map((v) => (
        <VoiceCard
          key={v.voice_id}
          voice={v}
          onPlaySample={onPlaySample}
          onAssign={onAssign}
        />
      ))}
    </div>
  );
}
