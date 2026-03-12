import { EmptyState } from "../shared";
import VoiceSamplePlayer from "./VoiceSamplePlayer";
import VoiceMetadataCard from "./VoiceMetadataCard";
import VoiceAssignmentPanel from "./VoiceAssignmentPanel";

/**
 * Right column: selected voice detail, sample player, metadata, assignment.
 * Renders empty state when no voice selected.
 */
export default function VoiceDetailPanel({
  voice,
  authFetch: _authFetch,
  npcOptions = [],
  onPlaySample,
  isPlayingSample,
  onAssignToNpc,
  onUnassignNpc,
  onReuseForNarration,
}) {
  const voiceId = voice?.voice_id || voice?.id;

  if (!voice) {
    return (
      <div className="flex flex-col min-h-0 flex-1 items-center justify-center p-4">
        <div className="w-full max-w-sm">
          <EmptyState message="Select a voice from the library to preview and manage it." />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-0 flex-1 gap-3 overflow-auto">
      <div className="shrink-0">
        <h2 className="font-heading text-[var(--gold)] text-lg">Voice console</h2>
        <div className="h-px bg-gradient-to-r from-[var(--gold)]/60 to-transparent mt-1" aria-hidden />
      </div>
      <div className="parchment rounded border border-[#a17a42] flex-1 min-h-0 overflow-auto p-3 space-y-4">
        <div className="shrink-0 pb-1 border-b border-[#c4a46b]/40">
          <p
            className="text-[10px] font-heading uppercase tracking-widest mb-0.5"
            style={{ color: "#7a4e18" }}
          >
            Selected Voice
          </p>
          <p className="font-heading text-lg leading-tight" style={{ color: "#1e0f06" }}>
            {voice.name}
          </p>
        </div>

        <div className="border-t border-[#5c3e23] pt-3">
          <VoiceSamplePlayer
            voiceId={voiceId}
            onPlay={onPlaySample}
            isPlaying={isPlayingSample}
            disabled={!voiceId}
          />
        </div>

        <div className="border-t border-[#5c3e23] pt-3">
          <VoiceMetadataCard voice={voice} />
        </div>

        <div className="border-t border-[#5c3e23] pt-3">
          <VoiceAssignmentPanel
            voice={voice}
            npcOptions={npcOptions}
            onAssign={onAssignToNpc}
            onUnassign={onUnassignNpc}
            disabled={!voiceId}
          />
        </div>

        {onReuseForNarration && (
          <div className="border-t border-[#5c3e23] pt-3">
            <p className="text-xs font-heading uppercase tracking-wider mb-1" style={{ color: "#6b3e10" }}>Actions</p>
            <button
              type="button"
              className="text-sm font-heading hover:underline"
              style={{ color: "#7a4010" }}
              onClick={() => onReuseForNarration(voice)}
            >
              Reuse for narration →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
