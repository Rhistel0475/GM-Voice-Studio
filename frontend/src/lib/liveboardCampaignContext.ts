/**
 * LiveBoard campaign context integration — connect LiveBoard UI to shared Campaign Context store.
 * Panels: Active Scene, NPC Presence, Codex Quick Reference, Session Log, Narration Control.
 * Do not redesign the approved LiveBoard layout.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import { getActionLogForActiveScene } from "../store/selectors";
import { getAiContext, buildAiContext } from "./aiContext";
import { narrateCurrentScene } from "./aiNarration";
import type { ActionLogEventType } from "../types";

/** Set the active scene (updates store; NPC presence and codex refs update via selectors). */
export function setActiveScene(sceneId: string | null): void {
  useCampaignContextStore.getState().setActiveScene(sceneId);
}

/** Build AI context from current store state (for narration, dialogue, GM assist). */
export { buildAiContext, getAiContext };

/** Generate scene narration using active scene and selected voice; returns text and optional clip. */
export async function generateSceneNarration(options?: {
  apiKey?: string;
  textOverride?: string;
  voiceId?: string;
}) {
  return narrateCurrentScene({
    apiKey: options?.apiKey,
    textOverride: options?.textOverride,
    voice: options?.voiceId ? { id: options.voiceId } : undefined,
  });
}

/** Add a session log entry (narration, player action, NPC dialogue, system, gm_note). */
export function addSessionLogEntry(entry: {
  type: ActionLogEventType;
  text: string;
  sessionId?: string;
  sceneId?: string;
}): void {
  useCampaignContextStore.getState().addActionLogEvent({
    type: entry.type,
    text: entry.text,
    sessionId: entry.sessionId,
    sceneId: entry.sceneId,
  });
}

/** Get the session log for the active scene (for Session Log panel). */
export function getSessionLog(): ReturnType<typeof getActionLogForActiveScene> {
  const state = useCampaignContextStore.getState();
  return getActionLogForActiveScene(state);
}

/** Push a codex entry to the active scene (link entry to current scene). */
export function pushCodexEntryToScene(entryId: string): void {
  const state = useCampaignContextStore.getState();
  const sceneId = state.activeSceneId;
  if (sceneId) {
    useCampaignContextStore.getState().addCodexEntryToScene(sceneId, entryId);
  }
}
