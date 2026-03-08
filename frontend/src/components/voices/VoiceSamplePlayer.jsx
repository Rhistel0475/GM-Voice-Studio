import React, { useState } from "react";
import { Play, Pause } from "lucide-react";

const DEFAULT_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog.";

/**
 * Sample playback: prominent play control, then sample text. Console-style; glow when playing.
 */
export default function VoiceSamplePlayer({
  voiceId,
  sampleText = DEFAULT_SAMPLE_TEXT,
  onSampleTextChange,
  onPlay,
  isPlaying,
  disabled,
}) {
  const [localText, setLocalText] = useState(sampleText);
  const text = onSampleTextChange ? sampleText : localText;
  const setText = onSampleTextChange || setLocalText;

  const handlePlay = () => {
    if (!voiceId || disabled) return;
    onPlay?.(voiceId, text?.trim() || DEFAULT_SAMPLE_TEXT);
  };

  return (
    <div
      className={`flex flex-col gap-3 rounded border p-3 transition-all ${
        isPlaying
          ? "border-[var(--gold)] ring-1 ring-[var(--gold)]/50 bg-[rgba(202,167,75,0.08)] shadow-[0_0_12px_rgba(202,167,75,0.2)]"
          : "border-[#734f2c] bg-[rgba(0,0,0,0.06)]"
      }`}
    >
      <p className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">Preview</p>
      <div className="flex flex-col sm:flex-row gap-3 items-start">
        <button
          type="button"
          className="flex items-center justify-center w-12 h-12 rounded-full border-2 border-[var(--gold)] bg-[rgba(202,167,75,0.12)] text-[var(--gold)] hover:bg-[rgba(202,167,75,0.2)] hover:shadow-[0_0_14px_rgba(202,167,75,0.35)] disabled:opacity-50 disabled:pointer-events-none shrink-0 transition-all"
          onClick={handlePlay}
          disabled={!voiceId || disabled}
          title={isPlaying ? "Playing…" : "Play sample"}
        >
          {isPlaying ? <Pause size={22} /> : <Play size={22} className="ml-0.5" />}
        </button>
        <div className="flex-1 w-full min-w-0">
          <textarea
            className="chat-input w-full min-h-[72px] resize-y text-sm"
            placeholder="Text to speak…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={disabled}
          />
        </div>
      </div>
      <p className="text-xs text-[var(--text-2)]">
        {isPlaying ? "Playing…" : "Enter text and press play to hear this voice."}
      </p>
    </div>
  );
}
