/**
 * Voice status (ready, training, etc.).
 */
export type VoiceStatus = "ready" | "training" | "failed" | "draft";

/**
 * Voice model — TTS voice with optional campaign scope and NPC assignments.
 */
export interface Voice {
  id: string;
  campaignId?: string;
  name: string;
  tone?: string;
  accent?: string;
  tags: string[];
  assignedNpcIds: string[];
  status?: VoiceStatus;
}
