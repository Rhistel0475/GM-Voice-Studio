/**
 * Scene state engine — runtime gameplay control: active scene, transitions, encounter launch,
 * NPC state, environment state, narration refresh triggers.
 */

import { create } from "zustand";
import { useCampaignContextStore } from "../store/campaignContext";
import { startEncounter as startEncounterManager, endEncounter as endEncounterManager } from "./encounterManager";
import type { Scene } from "../types";
import type { ExtractedEncounter } from "../types";

export type EnvironmentState = "normal" | "combat" | "social" | "exploration" | "tension";

export type NpcRuntimeState = "idle" | "speaking" | "hostile" | "friendly" | "hidden";

export interface SceneEvent {
  type: "transition" | "encounter_start" | "encounter_end" | "npc_state" | "environment" | "narration_trigger";
  sceneId?: string;
  payload?: unknown;
  at: string;
}

export interface SceneStateSnapshot {
  scene: Scene | null;
  sceneId: string | null;
  npcStates: Record<string, NpcRuntimeState>;
  environmentState: EnvironmentState;
  lastNarrationTriggerAt: string | null;
  recentEvents: SceneEvent[];
}

const MAX_RECENT_EVENTS = 50;

const defaultSnapshot: SceneStateSnapshot = {
  scene: null,
  sceneId: null,
  npcStates: {},
  environmentState: "normal",
  lastNarrationTriggerAt: null,
  recentEvents: [],
};

/** Encounter id -> ExtractedEncounter; app can register encounters from parsed data. */
const encounterRegistry = new Map<string, ExtractedEncounter>();

export function registerEncounter(id: string, encounter: ExtractedEncounter): void {
  encounterRegistry.set(id, encounter);
}

export function getEncounter(encounterId: string): ExtractedEncounter | undefined {
  return encounterRegistry.get(encounterId);
}

export const useSceneStateStore = create<SceneStateSnapshot & {
  createSceneState: (scene: Scene) => void;
  setActiveScene: (sceneId: string | null) => void;
  setNpcState: (npcId: string, state: NpcRuntimeState) => void;
  launchEncounter: (encounterId: string) => void;
  setEnvironmentState: (state: EnvironmentState) => void;
  suggestNarrationFromState: () => void;
  recordSceneEvent: (event: Omit<SceneEvent, "at">) => void;
}>((set, get) => ({
  ...defaultSnapshot,

  createSceneState(scene) {
    const state = useCampaignContextStore.getState();
    state.setActiveScene(scene.id);
    set({
      scene,
      sceneId: scene.id,
      npcStates: {},
      environmentState: "normal",
      lastNarrationTriggerAt: null,
      recentEvents: [...get().recentEvents, { type: "transition", sceneId: scene.id, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    });
  },

  setActiveScene(sceneId) {
    if (!sceneId) {
      set({
        scene: null,
        sceneId: null,
        npcStates: {},
        recentEvents: [...get().recentEvents, { type: "transition", at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
      });
      useCampaignContextStore.getState().setActiveScene(null);
      return;
    }
    const state = useCampaignContextStore.getState();
    const scene = state.scenes.find((s) => s.id === sceneId) ?? null;
    useCampaignContextStore.getState().setActiveScene(sceneId);
    set({
      scene,
      sceneId,
      npcStates: {},
      recentEvents: [...get().recentEvents, { type: "transition", sceneId, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    });
  },

  setNpcState(npcId, state) {
    set((s) => ({
      npcStates: { ...s.npcStates, [npcId]: state },
      recentEvents: [...s.recentEvents, { type: "npc_state", payload: { npcId, state }, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    }));
  },

  launchEncounter(encounterId) {
    const encounter = encounterRegistry.get(encounterId);
    if (!encounter) return;
    startEncounterManager(encounter);
    set((s) => ({
      environmentState: "combat",
      recentEvents: [...s.recentEvents, { type: "encounter_start", payload: { encounterId }, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    }));
  },

  setEnvironmentState(state) {
    set((s) => ({
      environmentState: state,
      recentEvents: [...s.recentEvents, { type: "environment", payload: { state }, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    }));
  },

  suggestNarrationFromState() {
    set((s) => ({
      lastNarrationTriggerAt: new Date().toISOString(),
      recentEvents: [...s.recentEvents, { type: "narration_trigger", at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    }));
  },

  recordSceneEvent(event) {
    set((s) => ({
      recentEvents: [...s.recentEvents, { ...event, at: new Date().toISOString() }].slice(-MAX_RECENT_EVENTS),
    }));
  },
}));

/** End active encounter and reset environment to normal. */
export function endEncounterFromSceneState(): void {
  endEncounterManager();
  useSceneStateStore.setState({ environmentState: "normal" });
}
