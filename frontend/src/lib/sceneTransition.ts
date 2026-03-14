/**
 * Scene transition controls — switch active scenes, update NPC presence/codex, trigger narration refresh.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import { useSceneStateStore } from "./sceneStateEngine";
import type { Scene, Npc, CodexEntry } from "../types";
import { getSceneNpcs, getSceneCodexEntries } from "../store/selectors";

/**
 * Set the active scene (updates store and scene state engine; NPC presence and codex refs update automatically).
 */
export function setActiveScene(sceneId: string | null): void {
  useSceneStateStore.getState().setActiveScene(sceneId);
}

/**
 * Get scenes available for the active campaign (for scene selector).
 */
export function getAvailableScenes(): Scene[] {
  const state = useCampaignContextStore.getState();
  const campaignId = state.activeCampaignId;
  if (!campaignId) return [];
  return state.scenes.filter((s) => s.campaignId === campaignId);
}

export interface ScenePreview {
  scene: Scene;
  npcs: Npc[];
  codexEntries: CodexEntry[];
}

/**
 * Preview a scene by id (scene + NPCs + codex for that scene). Does not switch active scene.
 */
export function previewScene(sceneId: string): ScenePreview | null {
  const state = useCampaignContextStore.getState();
  const scene = state.scenes.find((s) => s.id === sceneId) ?? null;
  if (!scene) return null;
  return {
    scene,
    npcs: getSceneNpcs(state, sceneId),
    codexEntries: getSceneCodexEntries(state, sceneId),
  };
}
