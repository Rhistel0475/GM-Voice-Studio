import { Upload } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * Step 1: Upload audio sample and optional name.
 */
export default function VoiceSampleUpload({
  file,
  onFileChange,
  name,
  onNameChange,
  onNext,
  disabled,
}) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-[var(--text-2)]">
        Upload a clear audio sample (WAV or MP3, 3–120 seconds). One speaker works best.
      </p>
      <label className="field-wrap">
        <span>Audio file</span>
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept="audio/*"
            className="chat-input flex-1"
            onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            disabled={disabled}
          />
          <Upload size={18} className="text-[var(--text-2)] shrink-0" />
        </div>
      </label>
      <label className="field-wrap">
        <span>Voice name (optional)</span>
        <input
          type="text"
          placeholder="e.g. Tavern Keeper"
          className="chat-input w-full"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          disabled={disabled}
        />
      </label>
      <FantasyButton
        variant="primary"
        className="w-full"
        onClick={onNext}
        disabled={!file || disabled}
      >
        Validate & start training
      </FantasyButton>
    </div>
  );
}
