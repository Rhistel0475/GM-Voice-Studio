import React from "react";
import { EmptyState } from "../shared";
import GeneratedAudioListItem from "./GeneratedAudioListItem";

/**
 * List of recent generated audio clips. Used in the library column.
 */
export default function GeneratedAudioList({
  clips = [],
  onPlayClip,
  playingClipId,
}) {
  if (!clips.length) {
    return (
      <div className="py-4">
        <EmptyState message="No generated audio yet. Use Narration or NPC dialogue to create clips." />
      </div>
    );
  }
  return (
    <div className="space-y-1.5 overflow-auto max-h-[200px]">
      {clips.map((clip) => (
        <GeneratedAudioListItem
          key={clip.id}
          clip={clip}
          onPlay={onPlayClip}
          isPlaying={playingClipId === clip.id}
        />
      ))}
    </div>
  );
}
