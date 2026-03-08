import React from "react";
import { StatusPill } from "../shared";

/**
 * Shown when there is a current audio blob/URL or status is loading/playing.
 */
export default function AudioPlaybackCard({ audioStatus, voiceName, onPlayPause, onDownload }) {
  if (audioStatus === "idle" && !voiceName) return null;

  return (
    <div className="rounded border border-[#5c3e23] bg-[#1a1008] p-2 flex items-center justify-between gap-2">
      <StatusPill status={audioStatus === "playing" ? "playing" : audioStatus === "loading" ? "generating" : "saved"} />
      {voiceName && <span className="text-xs text-[var(--text-2)]">{voiceName}</span>}
      <div className="flex gap-1">
        {onPlayPause && (
          <button type="button" className="send-btn text-xs px-2 py-1" onClick={onPlayPause}>
            {audioStatus === "playing" ? "Pause" : "Play"}
          </button>
        )}
        {onDownload && (
          <button type="button" className="cta-secondary text-xs" onClick={onDownload}>
            Download
          </button>
        )}
      </div>
    </div>
  );
}
