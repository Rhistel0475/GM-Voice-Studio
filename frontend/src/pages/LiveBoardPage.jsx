import { ScrollText } from "lucide-react";
import WorkspaceContainer from "../components/layout/WorkspaceContainer";
import GMControlPanel from "../components/live-board/GMControlPanel";
import SessionAssistantPanel from "../components/live-board/SessionAssistantPanel";
import { FantasyButton } from "../components/shared";
import { deriveNpcTone } from "../lib/sessionAssistant";

const BACKEND_URL = import.meta.env.DEV ? "http://localhost:7862" : "";

/**
 * Live Board — 3-column session console.
 * LEFT (3/12): GMControlPanel — Active Scene + Party
 * CENTER (6/12): Scene Stage or empty state
 * RIGHT (3/12): Session Assistant — max 2 suggestions
 */

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
      <div className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-3 xl:gap-4">

        {/* ── LEFT: Active Scene + Party ───────────────────────────── */}
        <aside className="xl:col-span-3 min-h-0 overflow-y-auto">
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
        <main className="xl:col-span-6 min-h-0 flex flex-col gap-3">

          {showSessionEmpty ? (
            /* Empty state — faded artwork background with dark vignette */
            <div
              className="flex-1 flex flex-col items-center justify-center rounded-lg overflow-hidden relative"
              style={{
                backgroundImage: `url('${BACKEND_URL}/static/img/gm_header.png')`,
                backgroundSize: "cover",
                backgroundPosition: "center 40%",
              }}
            >
              <div className="absolute inset-0" style={{ background: "rgba(8,4,2,0.82)" }} aria-hidden />
              <div className="relative z-10 flex flex-col items-center gap-4 py-14 text-center px-8">
                <div className="w-14 h-14 rounded-full border border-[#3a2510] bg-[#0f0804]/80 flex items-center justify-center">
                  <ScrollText size={26} className="text-[var(--gold)] opacity-50" aria-hidden />
                </div>
                <div>
                  <p className="font-heading text-[var(--text-1)] text-base tracking-widest uppercase">
                    Awaiting Your Command
                  </p>
                  <p className="mt-2 text-sm text-[var(--text-2)] leading-relaxed max-w-[260px] mx-auto">
                    Load a campaign and select a scene to begin your session.
                  </p>
                </div>
                <div className="flex flex-col gap-2 w-full max-w-[200px]">
                  {onNavigateToPrep ? (
                    <FantasyButton variant="secondary" className="text-xs w-full" onClick={onNavigateToPrep}>
                      Open Prep Room
                    </FantasyButton>
                  ) : null}
                </div>
              </div>
            </div>
          ) : (
            /* Scene Stage — read-aloud + NPC speak actions */
            scene && (
              <section className="panel-ornate scene-stage rounded-lg overflow-hidden flex-1">
                <div className="panel-head panel-head--row">
                  <div className="plaque flex items-center gap-3">
                    <span className="truncate">{scene.title || scene.name || "Scene"}</span>
                  </div>
                  {scene.location ? (
                    <span className="text-[10px] uppercase tracking-[0.16em] text-[#9b7440] flex-shrink-0 pr-2">
                      {scene.location}
                    </span>
                  ) : null}
                </div>
                <div className="panel-body space-y-4">
                  {readAloud ? (
                    <div className="read-aloud-text rounded-sm">
                      {readAloud}
                    </div>
                  ) : null}

                  {presentNpcs.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] flex items-center gap-2">
                        <span>NPCs Present</span>
                        <span className="text-[#4a3018]">—</span>
                        <span className="text-[#4a3018] normal-case not-italic">{presentNpcs.length}</span>
                      </div>
                      <div className="grid grid-cols-1 gap-1.5">
                        {presentNpcs.map((npc) => (
                          <article
                            key={npc.name}
                            className={`session-npc-card ${
                              selectedNpcName === npc.name ? "is-active" : ""
                            }`}
                          >
                            <div className="session-npc-card-head">
                              <div className="min-w-0 flex-1">
                                <div className="session-npc-card-name">{npc.name}</div>
                                <div className="session-npc-card-tone">{deriveNpcTone(npc)}</div>
                              </div>
                              <div className="session-npc-card-actions">
                                <FantasyButton
                                  variant="secondary"
                                  className="text-[10px] px-2 py-0.5"
                                  onClick={() => onSpeakNpcAction?.(npc)}
                                  disabled={!onSpeakNpcAction}
                                >
                                  Speak
                                </FantasyButton>
                                <FantasyButton
                                  variant="ghost"
                                  className="text-[10px] px-2 py-0.5"
                                  onClick={() => onWhisperNpc?.(npc)}
                                  disabled={!onWhisperNpc}
                                >
                                  Whisper
                                </FantasyButton>
                              </div>
                            </div>
                            {selectedNpcName === npc.name &&
                            (npc.description || npc.summary || npc.personality) ? (
                              <div className="session-npc-card-info">
                                {npc.description || npc.summary || npc.personality}
                              </div>
                            ) : null}
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
              suggestions={assistantSuggestions.slice(0, 2)}
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
