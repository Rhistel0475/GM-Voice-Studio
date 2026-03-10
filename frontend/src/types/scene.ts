/**
 * Scene model — location/encounter with linked NPCs, codex, logs, clips.
 */
export interface Scene {
  id: string;
  campaignId: string;
  sessionId?: string;
  title: string;
  summary: string;
  locationId?: string;
  npcIds: string[];
  codexEntryIds: string[];
  actionLogIds: string[];
  narrationClipIds: string[];
  tags?: string[];
}
