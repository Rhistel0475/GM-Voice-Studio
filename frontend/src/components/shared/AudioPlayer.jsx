import { Play, Pause } from "lucide-react";

/**
 * Shared audio player control: status label and play/pause. Stub for now;
 * wire to actual audio source (URL or blob) when used by Live Board / Voice Studio.
 */
export default function AudioPlayer({
  status = "idle",
  onPlay,
  onPause,
  disabled,
  className = "",
}) {
  const isPlaying = status === "playing";
  const label =
    status === "loading" ? "Loading…" : status === "playing" ? "Playing" : "Idle";

  return (
    <div
      className={`flex items-center gap-2 rounded-lg border border-[#5c3e23] bg-[#1a1008]/70 px-3 py-2 ${className}`.trim()}
      role="group"
      aria-label="Audio player"
    >
      <button
        type="button"
        className="p-1 rounded text-[var(--text-2)] hover:text-[var(--gold)] disabled:opacity-50 transition-colors"
        onClick={isPlaying ? onPause : onPlay}
        disabled={disabled}
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <span className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        {label}
      </span>
    </div>
  );
}
