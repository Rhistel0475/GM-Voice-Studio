import type { CampaignContextState } from "../store/campaignContext";
import type {
  Campaign,
  Session,
  Scene,
  Npc,
  CodexEntry,
  ActionLogEvent,
} from "../types";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  getActiveCampaign,
  getActiveSession,
  getActiveScene,
  getSceneNpcs,
  getSceneCodexEntries,
  getActionLogForActiveScene,
} from "../store/selectors";

/** Resolved location for AI context (from scene + optional codex entry). */
export interface AiContextLocation {
  id: string;
  name: string;
  summary?: string;
}

/**
 * Structured context for AI prompts (narration, dialogue, summaries).
 * Built from the shared campaign store; safe to pass to LLM prompt builders.
 */
export interface AiContext {
  campaign: Campaign | null;
  session: Session | null;
  scene: Scene | null;
  npcs: Npc[];
  location: AiContextLocation | null;
  recentEvents: ActionLogEvent[];
  codexReferences: CodexEntry[];
}

export interface BuildAiContextOptions {
  /** Max number of recent action log events to include (default 15). */
  maxRecentEvents?: number;
}

/**
 * Build a structured AI context from the campaign store state.
 * Use for scene narration, NPC dialogue, session summaries, event explanations.
 *
 * @param state - Current campaign context state (e.g. useCampaignContextStore.getState())
 * @param options - Optional limits (e.g. maxRecentEvents)
 * @returns Context object ready to be serialized into AI prompts
 */
export function buildAiContext(
  state: CampaignContextState,
  options: BuildAiContextOptions = {}
): AiContext {
  const { maxRecentEvents = 15 } = options;

  const campaign = getActiveCampaign(state);
  const session = getActiveSession(state);
  const scene = getActiveScene(state);

  const npcs = scene ? getSceneNpcs(state, scene.id) : [];
  const codexReferences = scene ? getSceneCodexEntries(state, scene.id) : [];
  const actionLogForScene = getActionLogForActiveScene(state);
  const recentEvents = actionLogForScene.slice(-maxRecentEvents);

  let location: AiContextLocation | null = null;
  if (scene) {
    if (scene.locationId) {
      const locationEntry = state.codexEntries.find(
        (e) => e.id === scene.locationId && e.type === "location"
      );
      if (locationEntry) {
        location = {
          id: locationEntry.id,
          name: locationEntry.title,
          summary: locationEntry.summary,
        };
      } else {
        location = {
          id: scene.locationId,
          name: scene.title,
          summary: scene.summary,
        };
      }
    } else {
      location = {
        id: scene.id,
        name: scene.title,
        summary: scene.summary,
      };
    }
  }

  return {
    campaign,
    session,
    scene,
    npcs,
    location,
    recentEvents,
    codexReferences,
  };
}

/**
 * Get current AI context from the store (for use outside React, e.g. in callbacks).
 * Uses current store state at call time.
 */
export function getAiContext(options?: BuildAiContextOptions): AiContext {
  const state = useCampaignContextStore.getState();
  return buildAiContext(state, options ?? {});
}

/**
 * React hook: subscribe to store and return current AI context.
 * Use when building context for a UI that should update when campaign state changes.
 */
export function useAiContext(options?: BuildAiContextOptions): AiContext {
  const state = useCampaignContextStore((s) => s);
  return buildAiContext(state, options ?? {});
}
