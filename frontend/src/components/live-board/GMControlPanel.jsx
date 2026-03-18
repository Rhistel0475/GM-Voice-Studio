/**
 * Left-column GM panel: Active Scene + Party Roster.
 *
 * Active Scene actions:
 *   PRIMARY  — Narrate (read aloud), Add Twist
 *   SECONDARY — Expand description (compact text link)
 *
 * Removed (not hidden): SceneControlPanel, EncounterLaunchPanel, SceneSuggestionsPanel
 */
export default function GMControlPanel({
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
}) {
  const party = campaignData?.party || [];
  const allNpcs = campaignData?.npcs || [];
  const sceneNpcs = (scene?.npcs || [])
    .map((name) => allNpcs.find((n) => n.name === name))
    .filter(Boolean);

  return (
    <div className="flex flex-col gap-3">

      {/* ── Active Scene ───────────────────────────────────────────── */}
      <section className="panel-ornate rounded-lg overflow-hidden">
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

              {/* Primary actions — 2 columns */}
              <div className="grid grid-cols-2 gap-1.5 pt-0.5">
                <button
                  type="button"
                  className="scene-trigger-btn"
                  onClick={() => onNarrateScene?.(scene)}
                  disabled={!onNarrateScene || isNarratingScene}
                >
                  <span>{isNarratingScene ? "Working…" : "Narrate"}</span>
                  <span className="scene-trigger-type">read aloud</span>
                </button>
                <button
                  type="button"
                  className="scene-trigger-btn"
                  onClick={() => onAddSceneTwist?.()}
                  disabled={!onAddSceneTwist || sceneActionBusy === "twist"}
                >
                  <span>{sceneActionBusy === "twist" ? "Working…" : "Add Twist"}</span>
                  <span className="scene-trigger-type">twist</span>
                </button>
              </div>

              {/* Secondary action — compact text link */}
              <button
                type="button"
                className="block w-full text-center text-[10px] text-[#7a5a30] hover:text-[#c8a050] transition-colors disabled:opacity-40"
                onClick={() => onExpandSceneDescription?.()}
                disabled={!onExpandSceneDescription || sceneActionBusy === "expand"}
              >
                {sceneActionBusy === "expand" ? "Expanding…" : "Expand description"}
              </button>

              {narrateSceneError ? (
                <div className="text-[10px] text-red-400">{narrateSceneError}</div>
              ) : null}
              {sceneActionError ? (
                <div className="text-[10px] text-red-400">{sceneActionError}</div>
              ) : null}

              {/* NPC selector rows */}
              {sceneNpcs.length > 0 ? (
                <div className="pt-1 border-t border-[#3a2510] space-y-1">
                  <div className="text-[10px] uppercase tracking-[0.13em] text-[var(--text-2)]">NPCs</div>
                  {sceneNpcs.map((npc) => (
                    <button
                      key={npc.name}
                      type="button"
                      className={`tracker-row w-full text-left cursor-pointer ${
                        selectedNpcName === npc.name ? "is-active border-[#d4af37]" : ""
                      }`}
                      onClick={() => onSelectNpc?.(selectedNpcName === npc.name ? null : npc.name)}
                    >
                      <span className="encounter-name text-xs">{npc.name}</span>
                      {npc.role ? (
                        <span className="text-[#9b7440] text-[10px]">{npc.role}</span>
                      ) : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <div className="text-xs text-[var(--text-2)] italic">No scene loaded.</div>
          )}
        </div>
      </section>

      {/* ── Party Roster ───────────────────────────────────────────── */}
      {party.length > 0 && (
        <section className="panel-ornate rounded-lg overflow-hidden flex-shrink-0">
          <div className="panel-head">
            <div className="plaque">Party</div>
          </div>
          <div className="panel-body space-y-1">
            {party.slice(0, 4).map((char) => (
              <div
                key={char.name}
                className="flex items-center justify-between text-xs py-0.5"
              >
                <span className="font-heading text-[var(--text-1)] text-[11px] tracking-wide">
                  {char.name}
                </span>
                <span className="text-[#9b7440] text-[10px] tabular-nums">
                  {char.hp !== "—" ? `HP ${char.hp}` : "—"}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

    </div>
  );
}
