/**
 * Action log event type (player, npc, narration, system).
 */
export type ActionLogEventType = "player" | "npc" | "narration" | "system";

/**
 * One entry in the session action log.
 * Backend: POST /api/session/events or websocket.
 */
export interface ActionLogEvent {
  id: string;
  sessionId: string;
  sceneId?: string;
  type: ActionLogEventType;
  text: string;
  createdAt: string;
}

/**
 * Narration clip — TTS output for playback/list.
 */
export interface NarrationClip {
  id: string;
  campaignId: string;
  sessionId?: string;
  sceneId?: string;
  title: string;
  voiceId?: string;
  audioUrl?: string;
  duration?: number;
  createdAt?: string;
}
