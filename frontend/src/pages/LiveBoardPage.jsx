import { LayoutDashboard } from "lucide-react";
import WorkspaceContainer from "../components/layout/WorkspaceContainer";
import GMControlPanel from "../components/live-board/GMControlPanel";
import SessionAssistantPanel from "../components/live-board/SessionAssistantPanel";
import { FantasyButton } from "../components/shared";
import { deriveNpcTone } from "../lib/sessionAssistant";

/**
 * Live Board — 3-column session console.
 *
 * LEFT   (2/12): GMControlPanel — Active Scene + Party
 * CENTER (7/12): Scene Stage (read-aloud + NPC cards) or compact empty state
 * RIGHT  (3/12): Session Assistant (compact, max 3 suggestions)
 *
 * Legacy panels not mounted here:
 *   SceneControlPanel, EncounterLaunchPanel, SceneSuggestionsPanel,
 *   StartSessionPanel, QuickTools, SessionAssistantSidebar
 */

console.log("LiveBoard minimal layout active");

export default function LiveBoardPage({
  campaignData,
  scene,
  selectedNpcName,
  onSelectNpc,
  onNarrateScene,
  isNarratingScene = false,
  narrateSceneError = "",
  onExpandSceneDescription,
  onAddSceneTwist,
  sceneActionBusy = "",
  sceneActionError = "",
  onSpeakNpcAction,
  onWhisperNpc,
  assistantSupported = false,
  assistantListening = false,
  assistantAnalyzing = false,
  assistantError = "",
  assistantPartialTranscript = "",
  assistantSuggestions = [],
  assistantContext = null,
  actionLog = [],
  assistantActionBusyId = "",
  onStartAssistantListening,
  onStopAssistantListening,
  onAnalyzeAssistant,
  onRunAssistantSuggestion,
  onNarrateAssistantSuggestion,
  onIgnoreAssistantSuggestion,
  showSessionEmpty = false,
  onNavigateToPrep,
}) {
  const allNpcs = campaignData?.npcs || [];
  const sceneNpcNames = scene?.npcs || [];
  const presentNpcs = sceneNpcNames
    .map((name) => allNpcs.find((n) => n.name === name))
    .filter(Boolean);

  const readAloud = String(
    scene?.read_aloud || scene?.notes || scene?.summary || ""
  ).trim();

  return (
    <WorkspaceContainer className="live-board">
      <div className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-4 xl:gap-5">

        {/* ── LEFT: Active Scene + Party ───────────────────────────── */}
        <aside className="xl:col-span-2 min-h-0 overflow-y-auto">
          <GMControlPanel
            campaignData={campaignData}
            scene={scene}
            selectedNpcName={selectedNpcName}
            onSelectNpc={onSelectNpc}
            onNarrateScene={onNarrateScene}
            isNarratingScene={isNarratingScene}
            narrateSceneError={narrateSceneError}
            onExpandSceneDescription={onExpandSceneDescription}
            onAddSceneTwist={onAddSceneTwist}
            sceneActionBusy={sceneActionBusy}
            sceneActionError={sceneActionError}
          />
        </aside>

        {/* ── CENTER: Scene Stage or empty state ──────────────────── */}
        <main className="xl:col-span-7 min-h-0 flex flex-col gap-3">

          {showSessionEmpty ? (
            /* Compact no-session state — replaces StartSessionPanel */
            <div className="flex-1 flex flex-col items-center justify-center gap-3 py-16 text-center">
              <LayoutDashboard size={28} className="text-[var(--gold)]/40" aria-hidden />
              <div>
                <p className="font-heading text-[var(--text-1)] text-sm tracking-wide">
                  No active scene loaded
                </p>
                <p className="mt-1 text-xs text-[var(--text-2)]">
                  Load a campaign in Prep or Campaign mode to begin
                </p>
              </div>
              {onNavigateToPrep ? (
                <FantasyButton variant="secondary" className="text-xs mt-1" onClick={onNavigateToPrep}>
                  Open Prep Room
                </FantasyButton>
              ) : null}
            </div>
          ) : (
            /* Scene Stage — read-aloud + NPC speak actions */
            scene && (
              <section className="panel-ornate rounded-lg overflow-hidden">
                <div className="panel-head">
                  <div className="plaque flex items-center gap-3">
                    <span className="truncate">{scene.title || scene.name || "Scene"}</span>
                    {scene.location ? (
                      <span className="text-[10px] font-normal uppercase tracking-[0.14em] text-[#9b7440] ml-auto flex-shrink-0">
                        {scene.location}
                      </span>
                    ) : null}
                  </div>
                </div>
                <div className="panel-body space-y-3">
                  {readAloud ? (
                    <p className="text-sm text-[var(--text-1)] leading-relaxed italic border-l-2 border-[var(--gold)]/40 pl-3">
                      {readAloud}
                    </p>
                  ) : null}

                  {presentNpcs.length > 0 && (
                    <div className="space-y-1.5">
                      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)]">
                        NPCs Present
                      </div>
                      <div className="flex flex-col gap-1.5">
                        {presentNpcs.map((npc) => (
                          <article
                            key={npc.name}
                            className={`session-npc-card ${
                              selectedNpcName === npc.name ? "is-active border-[#d4af37]" : ""
                            }`}
                          >
                            <div className="session-npc-card-head">
                              <div className="min-w-0">
                                <div className="session-npc-card-name">{npc.name}</div>
                                <div className="session-npc-card-tone">{deriveNpcTone(npc)}</div>
                              </div>
                            </div>
                            {selectedNpcName === npc.name &&
                            (npc.description || npc.summary || npc.personality) ? (
                              <div className="session-npc-card-info">
                                {npc.description || npc.summary || npc.personality}
                              </div>
                            ) : null}
                            <div className="session-npc-card-actions">
                              <FantasyButton
                                variant="secondary"
                                className="text-[10px] px-2 py-1"
                                onClick={() => onSpeakNpcAction?.(npc)}
                                disabled={!onSpeakNpcAction}
                              >
                                Speak
                              </FantasyButton>
                              <FantasyButton
                                variant="ghost"
                                className="text-[10px] px-2 py-1"
                                onClick={() => onWhisperNpc?.(npc)}
                                disabled={!onWhisperNpc}
                              >
                                Whisper
                              </FantasyButton>
                              <FantasyButton
                                variant={selectedNpcName === npc.name ? "primary" : "ghost"}
                                className="text-[10px] px-2 py-1"
                                onClick={() =>
                                  onSelectNpc?.(selectedNpcName === npc.name ? null : npc.name)
                                }
                              >
                                Info
                              </FantasyButton>
                            </div>
                          </article>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </section>
            )
          )}

        </main>

        {/* ── RIGHT: Session Assistant ─────────────────────────────── */}
        <aside className="xl:col-span-3 min-h-0 flex flex-col">
          <div className="min-h-0 flex-1 overflow-hidden">
            <SessionAssistantPanel
              compact
              supported={assistantSupported}
              listening={assistantListening}
              analyzing={assistantAnalyzing}
              error={assistantError}
              partialTranscript={assistantPartialTranscript}
              suggestions={assistantSuggestions.slice(0, 3)}
              context={assistantContext}
              sessionLog={actionLog}
              actionBusyId={assistantActionBusyId}
              onStartListening={onStartAssistantListening}
              onStopListening={onStopAssistantListening}
              onAnalyzeNow={onAnalyzeAssistant}
              onRunSuggestion={onRunAssistantSuggestion}
              onNarrateSuggestion={onNarrateAssistantSuggestion}
              onIgnoreSuggestion={onIgnoreAssistantSuggestion}
            />
          </div>
        </aside>

      </div>
    </WorkspaceContainer>
  );
}
