/**
 * Voice section in dossier: dropdown + play. Inline section, no card wrapper.
 */
import NPCVoiceAssignment from "./NPCVoiceAssignment";

export default function NPCVoiceAssignmentCard({
  voices = [],
  selectedVoiceId,
  onVoiceChange,
  onPlaySample,
  playing,
}) {
  return (
    <div className="border-b border-[#5c3e23] pb-3 last:border-0">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider mb-2">
        Assigned voice
      </div>
      <NPCVoiceAssignment
        voices={voices}
        selectedVoiceId={selectedVoiceId}
        onVoiceChange={onVoiceChange}
        onPlaySample={onPlaySample}
        playing={playing}
      />
    </div>
  );
}
