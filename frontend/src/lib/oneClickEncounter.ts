/**
 * One-click encounter launch from LiveBoard — activate encounter, set environment to combat,
 * populate enemies, add session log event, suggest narration update.
 */

import { useEncounterManagerStore, startEncounter } from "./encounterManager";
import { getEncounter, useSceneStateStore } from "./sceneStateEngine";
import { addSessionLogEntry } from "./sessionLogger";

/**
 * Launch an encounter by id: activate encounter state, set environment to combat,
 * populate enemies, add session log entry, trigger narration suggestion.
 */
export function launchEncounter(encounterId: string): void {
  const encounter = getEncounter(encounterId);
  if (!encounter) return;

  startEncounter(encounter);
  useSceneStateStore.setState({ environmentState: "combat" });
  addSessionLogEntry({
    type: "system",
    text: `Encounter started: ${encounter.name}.`,
  });
  useSceneStateStore.getState().suggestNarrationFromState();
}

/**
 * Preload encounter context (encounter + default enemy names) for UI without starting.
 */
export function preloadEncounterContext(encounterId: string): {
  encounter: ReturnType<typeof getEncounter>;
  enemyNames: string[];
} {
  const encounter = getEncounter(encounterId);
  if (!encounter) {
    return { encounter: undefined, enemyNames: [] };
  }
  const enemyNames = (encounter.enemyNpcRefs ?? []).map((r) => r.refName);
  return { encounter, enemyNames };
}

/**
 * End the current encounter, reset environment, add log entry.
 */
export function endEncounter(encounterId: string): void {
  const state = useEncounterManagerStore.getState();
  if (state.encounterId !== encounterId) return;

  useEncounterManagerStore.getState().endEncounter();
  useSceneStateStore.setState({ environmentState: "normal" });
  addSessionLogEntry({
    type: "system",
    text: "Encounter ended.",
  });
}
