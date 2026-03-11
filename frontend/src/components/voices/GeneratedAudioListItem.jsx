import { Play, FileAudio } from "lucide-react";

const TYPE_LABELS = {
  narration: "Narration",
  "npc-dialogue": "NPC",
  "scene-readout": "Scene",
};

/**
 * Single generated audio clip row: title, type, duration, play.
 */
export default function GeneratedAudioListItem({ clip, onPlay, isPlaying }) {
  const title = clip?.title || "Untitled";
  const type = clip?.type || "narration";
  const duration = clip?.duration;
  const createdAt = clip?.createdAt
    ? new Date(clip.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : "";

  return (
    <button
      type="button"
      className={`w-full text-left flex items-center gap-2 p-2 rounded transition-colors border ${
        isPlaying
          ? "border-[var(--gold)] ring-1 ring-[var(--gold)] bg-[#1a1008]"
          : "border-[#5c3e23] bg-[#1a1008] hover:border-[var(--gold)]"
      }`}
      onClick={() => onPlay?.(clip)}
      disabled={!onPlay}
      title="Play"
    >
      <Play size={14} className={`shrink-0 ${isPlaying ? "text-[var(--gold)]" : "text-[var(--text-2)]"}`} aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-heading text-[var(--text-1)] truncate">{title}</p>
        <div className="flex items-center gap-2 text-xs text-[var(--text-2)]">
          <span className="shrink-0">{TYPE_LABELS[type] || type}</span>
          {duration && <span>{duration}</span>}
          {createdAt && <span>{createdAt}</span>}
        </div>
      </div>
      <FileAudio size={14} className="shrink-0 text-[var(--text-2)] opacity-70" aria-hidden />
    </button>
  );
}
