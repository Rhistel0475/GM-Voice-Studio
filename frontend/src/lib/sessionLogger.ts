/**
 * Real-time session logger — append log entries, filter, integrate with scene/dialogue/encounters.
 * Log types: narration, player action, NPC dialogue, system event, GM note.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import { getActionLogForActiveScene } from "../store/selectors";
import type { ActionLogEvent, ActionLogEventType } from "../types";

export type SessionLogEntryType = ActionLogEventType;

export interface SessionLogEntryInput {
  type: SessionLogEntryType;
  text: string;
  sessionId?: string;
  sceneId?: string;
}

/**
 * Add a session log entry (real-time append). Integrates with active scene/session.
 */
export function addSessionLogEntry(entry: SessionLogEntryInput): void {
  useCampaignContextStore.getState().addActionLogEvent({
    type: entry.type,
    text: entry.text,
    sessionId: entry.sessionId,
    sceneId: entry.sceneId,
  });
}

/**
 * Get the full session log for the active scene (for filtering in UI).
 */
export function getSessionLog(): ActionLogEvent[] {
  const state = useCampaignContextStore.getState();
  return getActionLogForActiveScene(state);
}

/**
 * Get the most recent session events (for previews, AI context, etc.).
 */
export function getRecentSessionEvents(limit: number): ActionLogEvent[] {
  const log = getSessionLog();
  return log.slice(-limit);
}
