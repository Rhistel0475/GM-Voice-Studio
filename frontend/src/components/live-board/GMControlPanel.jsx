import EncounterLaunchPanel from "./EncounterLaunchPanel";
import SceneControlPanel from "./SceneControlPanel";
import SceneSuggestionsPanel from "./SceneSuggestionsPanel";
import { getPartyPlaceholder } from "../../lib/placeholders";

export default function GMControlPanel({
  campaignData,
  scene,
  selectedNpcName,
  onSelectNpc,
  onTriggerScene,
  activeTriggerName = "",
  triggerSceneError = "",
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
  onExpandSceneDescription,
  onAddSceneTwist,
  sceneActionBusy = "",
  sceneActionError = "",
}) {
  const party = campaignData?.party || [];
  const allNpcs = campaignData?.npcs || [];
  const sceneNpcs = (scene?.npcs || []).map((npcName) => allNpcs.find((n) => n.name === npcName)).filter(Boolean);
  const allItems = campaignData?.items || [];
  const sceneItems = (scene?.items || []).map((itemName) => allItems.find((i) => i.name === itemName)).filter(Boolean);
  const allReveals = campaignData?.reveals || [];
  const sceneReveals = (scene?.reveals || []).map((revealName) => allReveals.find((r) => r.name === revealName)).filter(Boolean);

  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <SceneControlPanel
        scene={scene}
        onTrigger={onTriggerScene}
        busyTriggerName={activeTriggerName}
        triggerError={triggerSceneError}
      />

      <EncounterLaunchPanel
        scene={scene}
        onLaunch={onLaunchEncounter}
        isLaunching={isLaunchingEncounter}
        error={launchEncounterError}
      />

      <SceneSuggestionsPanel
        suggestions={sceneSuggestions}
        isLoading={sceneSuggestionsLoading}
        error={sceneSuggestionsError}
        onActivate={onActivateSuggestedScene}
        busySceneId={activeSuggestedSceneId}
      />

      <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
        <div className="panel-head">
          <div className="plaque">Active Scene</div>
        </div>
        <div className="panel-body space-y-2 overflow-y-auto min-h-[160px] max-h-[320px]">
          {scene ? (
            <div className="rounded border border-[#5c3e23] bg-[#1a1008]/70 p-2 space-y-2">
              <div className="font-heading text-sm text-[var(--text-1)]">
                {scene.title || scene.name || "Active Scene"}
              </div>
              {scene.location ? (
                <div className="text-[11px] uppercase tracking-[0.14em] text-[#9b7440]">
                  {scene.location}
                </div>
              ) : null}
              <div className="grid grid-cols-1 gap-1.5">
                <button
                  type="button"
                  className="scene-trigger-btn"
                  onClick={() => onNarrateScene?.(scene)}
                  disabled={!onNarrateScene || isNarratingScene}
                >
                  <span>{isNarratingScene ? "Working..." : "Read Aloud"}</span>
                  <span className="scene-trigger-type">narration</span>
                </button>
                <button
                  type="button"
                  className="scene-trigger-btn"
                  onClick={() => onExpandSceneDescription?.()}
                  disabled={!onExpandSceneDescription || sceneActionBusy === "expand"}
                >
                  <span>{sceneActionBusy === "expand" ? "Working..." : "Expand Description"}</span>
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
              {narrateSceneError ? <div className="text-xs text-red-400">{narrateSceneError}</div> : null}
              {sceneActionError ? <div className="text-xs text-red-400">{sceneActionError}</div> : null}
            </div>
          ) : null}
          {sceneNpcs.map((npc) => (
            <button
              key={npc.name}
              type="button"
              className={`tracker-row w-full text-left cursor-pointer ${selectedNpcName === npc.name ? "is-active border-[#d4af37]" : ""}`}
              onClick={() => onSelectNpc(npc.name)}
            >
              <span className="encounter-name">{npc.name}</span>
              <span className="text-[#9b7440] text-xs">{npc.role}</span>
            </button>
          ))}
          {sceneItems.map((item) => (
            <div key={item.name} className="tracker-row w-full text-left">
              <span className="encounter-name">{item.name}</span>
              <span className="text-[#9b7440] text-xs">Item</span>
            </div>
          ))}
          {sceneReveals.map((reveal) => (
            <div key={reveal.name} className="tracker-row w-full text-left">
              <span className="encounter-name">{reveal.name}</span>
              <span className="text-[#9b7440] text-xs capitalize">{reveal.type}</span>
            </div>
          ))}
          {sceneNpcs.length === 0 && sceneItems.length === 0 && sceneReveals.length === 0 && (
            <div className="intake-empty text-xs">No NPCs, items, or reveals in this scene.</div>
          )}
        </div>
      </section>

      {party.length > 0 && (
        <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden flex-shrink-0">
          <div className="panel-head">
            <div className="plaque">Party Roster</div>
          </div>
          <div className="panel-body">
            <div className="grid grid-cols-2 gap-1">
              {party.slice(0, 4).map((char) => {
                const portraitSrc = getPartyPlaceholder();
                return (
                  <article key={char.name} className="party-card">
                    <div
                      className="party-face"
                      style={{ backgroundImage: `url(${portraitSrc})`, backgroundSize: "cover", backgroundPosition: "center" }}
                    />
                    <div className="party-meta">
                      <div className="name text-sm">{char.name}</div>
                      <div className="stats text-xs">
                        {char.hp !== "—" ? (
                          <>HP {char.hp} <span>/ AC {char.ac}</span></>
                        ) : (
                          <span className="text-[#6b5030]">—</span>
                        )}
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
