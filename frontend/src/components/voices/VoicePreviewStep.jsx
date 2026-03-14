import { Play, ChevronLeft } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * Step 3: Preview the cloned voice before saving.
 */
export default function VoicePreviewStep({
  voiceId,
  voiceName,
  onPlayPreview,
  isPlaying,
  onNext,
  onBack,
  disabled,
}) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-[var(--ink-1)]">
        {voiceName ? `Preview: ${voiceName}` : "Preview your cloned voice."}
      </p>
      <FantasyButton
        variant="primary"
        className="w-full"
        onClick={() => onPlayPreview?.(voiceId)}
        disabled={!voiceId || disabled}
      >
        <Play size={14} className="inline mr-2" />
        {isPlaying ? "Playing…" : "Play preview"}
      </FantasyButton>
      <div className="flex gap-2">
        {onBack && (
          <FantasyButton variant="ghost" className="flex-1" onClick={onBack} disabled={disabled}>
            <ChevronLeft size={14} className="inline mr-1" /> Back
          </FantasyButton>
        )}
        <FantasyButton variant="secondary" className="flex-1" onClick={onNext} disabled={disabled}>
          Next: name and save
        </FantasyButton>
      </div>
    </div>
  );
}
