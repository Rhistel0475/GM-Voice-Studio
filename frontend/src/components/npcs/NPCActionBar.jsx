import { FantasyButton } from "../shared";
import { Sparkles, Save, Mic, Send } from "lucide-react";

/**
 * Action bar: grouped into Generate vs Save/Assign/Push. Clear labels and fantasy styling.
 */
export default function NPCActionBar({
  onGenerate,
  onRegeneratePersonality,
  onRegenerateBackstory,
  onSave,
  onSpeak,
  onAssignVoice,
  onPushToLiveBoard,
  generating = false,
  saving = false,
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Actions
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-2">
        <div className="flex flex-wrap gap-2 items-center">
          <FantasyButton
            variant="primary"
            onClick={onGenerate}
            disabled={generating}
          >
            {generating ? "Generating…" : (
              <>
                <Sparkles size={14} className="inline mr-1.5 shrink-0" />
                Generate NPC
              </>
            )}
          </FantasyButton>
          <FantasyButton variant="secondary" onClick={onRegeneratePersonality} disabled={generating} className="text-xs">
            Regenerate personality
          </FantasyButton>
          <FantasyButton variant="secondary" onClick={onRegenerateBackstory} className="text-xs">
            Regenerate backstory
          </FantasyButton>
        </div>
        <div className="h-px bg-[#5c3e23] sm:h-4 sm:w-px sm:h-auto" aria-hidden />
        <div className="flex flex-wrap gap-2 items-center">
          <FantasyButton variant="secondary" onClick={onSave} disabled={saving}>
            {saving ? "Saving…" : <><Save size={14} className="inline mr-1.5 shrink-0" />Save NPC</>}
          </FantasyButton>
          {onSpeak ? (
            <FantasyButton variant="secondary" onClick={onSpeak}>
              <Mic size={14} className="inline mr-1.5 shrink-0" />
              Speak
            </FantasyButton>
          ) : null}
          <FantasyButton variant="secondary" onClick={onAssignVoice}>
            <Mic size={14} className="inline mr-1.5 shrink-0" />
            Assign voice
          </FantasyButton>
          <FantasyButton variant="secondary" onClick={onPushToLiveBoard}>
            <Send size={14} className="inline mr-1.5 shrink-0" />
            Push to Live Board
          </FantasyButton>
        </div>
      </div>
    </div>
  );
}
