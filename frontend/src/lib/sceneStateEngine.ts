/**
 * One-click encounter launch from LiveBoard — activate encounter, set environment to combat,
 * populate enemies, add session log event, suggest narration update.
 */

import { useEncounterManagerStore } from "./encounterManager";
import { registerEncounter } from "./sceneStateEngine";
import { addSessionLogEntry } from "./sessionLogger";
import { useSceneStateStore } from "./sceneStateEngine";
import type { ExtractedEncounter } from "../types";

/**
 * Register an encounter so it can be launched by id. Call when loading parsed/imported encounters.
 */
export function registerEncounterForLaunch(id: string, encounter: ExtractedEncounter): void {
  registerEncounter(id, encounter);
}

/**
 * Preload encounter context (encounter + default actors) without starting. Returns data for UI.
 */
export function preloadEncounterContext(encounterId: string): {
  encounter: ExtractedEncounter | null;
  enemyNames: string[];
} {
  const encounter = useEncounterManagerStore.getState().encounter;
  if (encounter?.id === encounterId) {
    const actors = useEncounterManagerStore.getState().actors;
    return {
      encounter,
      enemyNames: actors.map((a) => a.name),
    };
  }
  const registered = useSceneStateStore.getState();
  const map = (useSceneStateStore as unknown as { getState: () => { _encounterRegistry?: Map<string, ExtractedEncounter> } }).getState();
  const registry = (sceneStateEngine as { registerEncounter: (id: string, e: ExtractedEncounter) => void; getEncounter?: (id: string) => ExtractedEncounter | undefined }).getEncounter
    ? undefined
    : undefined;
  const sceneStateEngine = require("./sceneStateEngine");
  const enc = (sceneStateEngine as { getEncounter?: (id: string) => ExtractedEncounter | undefined }).getEncounter?.(encounterId);
  if (!enc) {
    return { encounter: null, enemyNames: [] };
  }
  const enemyNames = (enc.enemyNpcRefs ?? []).map((r) => r.refName);
  return { encounter: enc, enemyNames };
}

I need to fix preloadEncounterContext - we don't have getEncounter in sceneStateEngine, we have registerEncounter and a private encounterRegistry Map. So preloadEncounterContext needs to get the encounter from somewhere. The registry is inside sceneStateEngine and not exported. So I'll export a getEncounter(id) from sceneStateEngine that returns registry.get(id), then use it in preloadEncounterContext.
</think>
Exporting `getEncounter` from the scene state engine and fixing the one-click encounter module.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace