import NPCPortraitCard from "./NPCPortraitCard";
import NPCSummaryCard from "./NPCSummaryCard";
import NPCDetailCard from "./NPCDetailCard";
import NPCVoiceAssignmentCard from "./NPCVoiceAssignmentCard";
import NPCActionBar from "./NPCActionBar";
import { EmptyState } from "../shared";

/**
 * Right panel: character dossier — portrait, summary, details, voice, actions.
 * profile = merged draft + selectedNpc for display.
 */
export default function NPCPreviewCard({
  profile,
  voices = [],
  selectedVoiceId,
  onVoiceChange,
  onPlaySample,
  playing,
  onGenerate,
  onRegeneratePersonality,
  onRegenerateBackstory,
  onSave,
  onSpeak,
  onAssignVoice,
  onPushToLiveBoard,
  generating = false,
  saving = false,
  suggestion,
  onApplySuggestion,
}) {
  const hasProfile = profile && (profile.name || profile.summary);

  return (
    <div className="flex flex-col gap-3 min-h-0">
      <div className="shrink-0">
        <h2 className="font-heading text-[var(--gold)] text-lg">Character dossier</h2>
        <div className="h-px bg-gradient-to-r from-[var(--gold)]/60 to-transparent mt-1" />
      </div>
      {!hasProfile ? (
        <EmptyState message="Create or select an NPC to see the preview." />
      ) : (
        <>
          <div className="shrink-0">
            <NPCPortraitCard npc={profile} />
          </div>
          <div className="parchment rounded border border-[#a17a42] flex-1 min-h-0 overflow-auto p-3 space-y-3">
            <NPCSummaryCard profile={profile} />
            <NPCDetailCard npc={profile} />
            <NPCVoiceAssignmentCard
              voices={voices}
              selectedVoiceId={selectedVoiceId}
              onVoiceChange={onVoiceChange}
              onPlaySample={onPlaySample}
              playing={playing}
              suggestion={suggestion}
              onApplySuggestion={onApplySuggestion}
            />
          </div>
          <div className="shrink-0 pt-2 border-t border-[#5c3e23]">
            <NPCActionBar
              onGenerate={onGenerate}
              onRegeneratePersonality={onRegeneratePersonality}
              onRegenerateBackstory={onRegenerateBackstory}
              onSave={onSave}
              onSpeak={onSpeak}
              onAssignVoice={onAssignVoice}
              onPushToLiveBoard={onPushToLiveBoard}
              generating={generating}
              saving={saving}
            />
          </div>
        </>
      )}
    </div>
  );
}
