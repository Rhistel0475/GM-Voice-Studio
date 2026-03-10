/**
 * NPC disposition toward the party.
 */
export type NpcDisposition = "friendly" | "neutral" | "hostile" | "unknown";

/**
 * NPC model — character in the campaign with optional voice assignment.
 */
export interface Npc {
  id: string;
  campaignId: string;
  name: string;
  summary: string;
  role?: string;
  profession?: string;
  locationId?: string;
  factionId?: string;
  voiceId?: string;
  tags: string[];
  disposition?: NpcDisposition;
}
