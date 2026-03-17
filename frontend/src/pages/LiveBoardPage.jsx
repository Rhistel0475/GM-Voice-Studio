import SectionHeader from "../components/layout/SectionHeader";
import WorkspaceContainer from "../components/layout/WorkspaceContainer";
import PanelShell from "../components/layout/PanelShell";
import GMControlPanel from "../components/live-board/GMControlPanel";
import SessionStream from "../components/live-board/SessionStream";
import CodexSidePanel from "../components/live-board/CodexSidePanel";
import { LiveBoardEmptyState } from "../components/shared";

/**
 * Live Board: central command center for the GM during live gameplay.
 * Three-zone layout: Left = GM Control (quick tools, scene, party) | Center = Session stream | Right = Codex.
 * Banner (campaign, scene, timer, audio) is provided by AppShell TopBanner.
 */
export default function LiveBoardPage({
  campaignData,
  scene,
  selectedNpcName,
  onSelectNpc,
  onInsertIntoNarration,
  authFetch,
  onNarrateCampaignAnswer,
  onSceneTrigger,
  activeSceneTriggerName = "",
  sceneTriggerError = "",
  sceneSuggestions = [],
  sceneSuggestionsLoading = false,
  sceneSuggestionsError = "",
  onActivateSuggestedScene,
  activeSuggestedSceneId = "",
  onLaunchEncounter,
  isLaunchingEncounter = false,
  launchEncounterError = "",
  onNarrateScene,
  isNarratingScene = false,
  narrateSceneError = "",
  middleColumn,
  showSessionEmpty = false,
  emptyStateContent = null,
}) {
  const npcs = campaignData?.npcs || [];
  const sceneNpcNames = scene?.npcs || [];
  const presentNpcs = sceneNpcNames
    .map((name) => npcs.find((n) => n.name === name))
    .filter(Boolean);

  return (
    <WorkspaceContainer className="live-board gap-5 xl:gap-6">
      <div className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-5 xl:gap-6">
        {/* LEFT: GM Tools — quick tools, active scene, party roster */}
        <aside className="xl:col-span-2 min-h-0 flex flex-col gap-3">
          <SectionHeader title="GM Control" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden">
            <GMControlPanel
              campaignData={campaignData}
              scene={scene}
              selectedNpcName={selectedNpcName}
              onSelectNpc={onSelectNpc}
              onTriggerScene={onSceneTrigger}
              activeTriggerName={activeSceneTriggerName}
              triggerSceneError={sceneTriggerError}
              sceneSuggestions={sceneSuggestions}
              sceneSuggestionsLoading={sceneSuggestionsLoading}
              sceneSuggestionsError={sceneSuggestionsError}
              onActivateSuggestedScene={onActivateSuggestedScene}
              activeSuggestedSceneId={activeSuggestedSceneId}
              onLaunchEncounter={onLaunchEncounter}
              isLaunchingEncounter={isLaunchingEncounter}
              launchEncounterError={launchEncounterError}
            />
          </div>
        </aside>

        {/* CENTER: Session stream — or empty state when no campaign/session */}
        <main className="xl:col-span-5 min-h-0 flex flex-col gap-3">
          <SectionHeader title="Live Session" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden flex flex-col items-center justify-center p-4">
            {showSessionEmpty ? (
              emptyStateContent || <LiveBoardEmptyState />
            ) : (
              <SessionStream>{middleColumn}</SessionStream>
            )}
          </div>
        </main>

        {/* NPC Presence — spotlight on who is in the current scene */}
        <aside className="xl:col-span-2 min-h-0 flex flex-col gap-3">
          <PanelShell title="NPC Presence" className="h-full min-h-0 flex flex-col rounded-lg overflow-hidden">
            <div className="panel-body min-h-0 flex-1 overflow-y-auto space-y-2">
              {presentNpcs.length > 0 ? (
                presentNpcs.map((npc) => (
                  <button
                    key={npc.name}
                    type="button"
                    className={`tracker-row w-full text-left cursor-pointer ${
                      selectedNpcName === npc.name ? "is-active border-[#d4af37]" : ""
                    }`}
                    onClick={() => onSelectNpc(npc.name)}
                  >
                    <span className="encounter-name">{npc.name}</span>
                    {npc.role && <span className="text-[#9b7440] text-xs">{npc.role}</span>}
                  </button>
                ))
              ) : (
                <div className="intake-empty text-xs">No NPCs are currently present in this scene.</div>
              )}
            </div>
          </PanelShell>
        </aside>

        {/* RIGHT: Codex — Documents, NPCs, Locations, Rules tabs + quick preview cards */}
        <aside className="xl:col-span-3 min-h-0 flex flex-col gap-3">
          <SectionHeader title="Codex" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden">
            <CodexSidePanel
              campaignData={campaignData}
              onInsertIntoNarration={onInsertIntoNarration}
              onSpeakNpc={onSelectNpc}
              authFetch={authFetch}
              onNarrateAnswer={onNarrateCampaignAnswer}
            />
          </div>
        </aside>
      </div>
    </WorkspaceContainer>
  );
}
