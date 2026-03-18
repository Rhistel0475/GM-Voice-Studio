import WorkspaceContainer from "../components/layout/WorkspaceContainer";
import SessionAssistantPanel from "../components/live-board/SessionAssistantPanel";
import { FantasyButton, LiveBoardEmptyState } from "../components/shared";
import { deriveNpcTone } from "../lib/sessionAssistant";
import { getPartyPlaceholder } from "../lib/placeholders";

/**
 * Live Board — minimal 3-column session console.
 *
 * LEFT   (2/12): Active Scene controls + Party Roster
 * CENTER (7/12): Scene stage (read-aloud + NPC speak actions) + session stream
 * RIGHT  (3/12): Session Assistant (compact, max 3 suggestions)
 *
 * Panels removed (not hidden — deleted):
 *   QuickTools, SceneControlPanel, EncounterLaunchPanel,
 *   SceneSuggestionsPanel, CampaignBrainPanel / CodexSidePanel,
 *   StartSessionPanel (shown only as emptyStateContent from App.jsx),
 *   GMControlPanel, SessionAssistantSidebar
 */

// Debug: confirm minimal layout is mounted
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
  emptyStateContent = null,
}) {
  const allNpcs = campaignData?.npcs || [];
  const party = campaignData?.party || [];
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

        {/* ── LEFT: Active Scene + Party ─────────────────────────── */}
        <aside className="xl:col-span-2 min-h-0 flex flex-col gap-3 overflow-y-auto">

          {/* Panel 1: Active Scene */}
          <section className="panel-ornate flex-shrink-0 rounded-lg overflow-hidden">
            <div className="panel-head">
              <div className="plaque">Active Scene</div>
            </div>
            <div className="panel-body space-y-2">
              {scene ? (
                <>
                  <div className="font-heading text-sm text-[var(--text-1)] leading-tight">
                    {scene.title || scene.name || "Scene"}
                  </div>
                  {scene.location ? (
                    <div className="text-[10px] uppercase tracking-[0.14em] text-[#9b7440]">
                      {scene.location}
                    </div>
                  ) : null}

                  <div className="flex flex-col gap-1.5 pt-1">
                    <button
                      type="button"
                      className="scene-trigger-btn"
                      onClick={() => onNarrateScene?.(scene)}
                      disabled={!onNarrateScene || isNarratingScene}
                    >
                      <span>{isNarratingScene ? "Working..." : "Narrate"}</span>
                      <span className="scene-trigger-type">read aloud</span>
                    </button>
                    <button
                      type="button"
                      className="scene-trigger-btn"
                      onClick={() => onExpandSceneDescription?.()}
                      disabled={!onExpandSceneDescription || sceneActionBusy === "expand"}
                    >
                      <span>{sceneActionBusy === "expand" ? "Working..." : "Expand"}</span>
                      <span className="scene-trigger-type">scene</span>
                    </button>
                    <button
                      type="button"
                      className="scene-trigger-btn"
                      onClick={() => onAddSceneTwist?.()}
                      disabled={!onAddSceneTwist || sceneActionBusy === "twist"}
                    >
                      <span>{sceneActionBusy === "twist" ? "Working..." : "Add Twist"}</span>
                      <span className="scene-trigger-type">twist</span>
                    </button>
                  </div>

                  {narrateSceneError ? (
                    <div className="text-[11px] text-red-400">{narrateSceneError}</div>
                  ) : null}
                  {sceneActionError ? (
                    <div className="text-[11px] text-red-400">{sceneActionError}</div>
                  ) : null}

                  {sceneNpcNames.length > 0 ? (
                    <div className="pt-1 border-t border-[#3a2510] space-y-1">
                      <div className="text-[10px] uppercase tracking-[0.13em] text-[var(--text-2)]">
                        NPCs
                      </div>
                      {presentNpcs.map((npc) => (
                        <button
                          key={npc.name}
                          type="button"
                          className={`tracker-row w-full text-left cursor-pointer ${
                            selectedNpcName === npc.name
                              ? "is-active border-[#d4af37]"
                              : ""
                          }`}
                          onClick={() =>
                            onSelectNpc?.(
                              selectedNpcName === npc.name ? null : npc.name
                            )
                          }
                        >
                          <span className="encounter-name text-xs">{npc.name}</span>
                        </button>
                      ))}
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="intake-empty text-xs">No active scene.</div>
              )}
            </div>
          </section>

          {/* Panel 2: Party Roster — only when party exists */}
          {party.length > 0 && (
            <section className="panel-ornate flex-shrink-0 rounded-lg overflow-hidden">
              <div className="panel-head">
                <div className="plaque">Party</div>
              </div>
              <div className="panel-body">
                <div className="grid grid-cols-2 gap-1">
                  {party.slice(0, 4).map((char) => (
                    <article key={char.name} className="party-card">
                      <div
                        className="party-face"
                        style={{
                          backgroundImage: `url(${getPartyPlaceholder()})`,
                          backgroundSize: "cover",
                          backgroundPosition: "center",
                        }}
                      />
                      <div className="party-meta">
                        <div className="name text-xs">{char.name}</div>
                        <div className="stats text-[10px]">
                          {char.hp !== "—" ? (
                            <>HP {char.hp}</>
                          ) : (
                            <span className="text-[#6b5030]">—</span>
                          )}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          )}

        </aside>

        {/* ── CENTER: Scene stage + Session stream ────────────────── */}
        <main className="xl:col-span-7 min-h-0 flex flex-col gap-3">

          {/* Panel 3: Scene Stage — read-aloud + NPC speak actions */}
          {scene && (
            <section className="panel-ornate flex-shrink-0 rounded-lg overflow-hidden">
              <div className="panel-head">
                <div className="plaque flex items-center gap-3">
                  <span className="truncate">
                    {scene.title || scene.name || "Scene"}
                  </span>
                  {scene.location ? (
                    <span className="text-[10px] font-normal uppercase tracking-[0.14em] text-[#9b7440] ml-auto flex-shrink-0">
                      {scene.location}
                    </span>
                  ) : null}
                </div>
              </div>
              <div className="panel-body space-y-3">
                {readAloud && (
                  <p className="text-sm text-[var(--text-1)] leading-relaxed italic border-l-2 border-[var(--gold)]/40 pl-3">
                    {readAloud}
                  </p>
                )}

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
                            selectedNpcName === npc.name
                              ? "is-active border-[#d4af37]"
                              : ""
                          }`}
                        >
                          <div className="session-npc-card-head">
                            <div className="min-w-0">
                              <div className="session-npc-card-name">{npc.name}</div>
                              <div className="session-npc-card-tone">
                                {deriveNpcTone(npc)}
                              </div>
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
                              variant={
                                selectedNpcName === npc.name ? "primary" : "ghost"
                              }
                              className="text-[10px] px-2 py-1"
                              onClick={() =>
                                onSelectNpc?.(
                                  selectedNpcName === npc.name ? null : npc.name
                                )
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
          )}

          {/* Empty state — shown before a session is started */}
          {showSessionEmpty && (
            <div className="min-h-0 flex-1 flex items-center justify-center p-4">
              {emptyStateContent || <LiveBoardEmptyState />}
            </div>
          )}

        </main>

        {/* ── RIGHT: Session Assistant (compact, max 3 suggestions) ── */}
        <aside className="xl:col-span-3 min-h-0 flex flex-col">
          {/* Panel 5: Session Assistant */}
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
