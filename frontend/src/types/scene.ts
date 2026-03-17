/**
 * Scene model — location/encounter with linked NPCs, codex, logs, clips.
 */
export type SceneTriggerType =
  | "narration"
  | "dialogue"
  | "ai_dialogue"
  | "ai_action"
  | "lore"
  | "quest_hook"
  | "text"
  | string;

export interface SceneTrigger {
  name: string;
  type: SceneTriggerType;
  text?: string;
  action?: string | Record<string, unknown>;
  npcId?: string;
  npcName?: string;
  voiceId?: string;
}

export interface SceneAmbienceTrack {
  track_id?: string;
  atmosphere_type?: string;
  label?: string;
  filename?: string;
  url?: string;
  loop?: boolean;
  volume?: number;
}

export interface Scene {
  id: string;
  campaignId: string;
  sessionId?: string;
  title: string;
  name?: string;
  summary: string;
  description?: string;
  act?: string;
  type?: string;
  locationId?: string;
  npcIds: string[];
  codexEntryIds: string[];
  actionLogIds: string[];
  narrationClipIds: string[];
  tags?: string[];
  location?: string;
  locationSummary?: string;
  notes?: string;
  readAloud?: string;
  read_aloud?: string;
  atmosphereType?: string;
  atmosphere_type?: string;
  ambienceTrack?: string | SceneAmbienceTrack | null;
  ambience_track?: string | SceneAmbienceTrack | null;
  narratorVoiceId?: string;
  narrator_voice_id?: string;
  triggers?: SceneTrigger[];
  connectedScenes?: string[];
  connected_scenes?: string[];
  relatedNpcNames?: string[];
  relatedLocationNames?: string[];
  relatedQuestNames?: string[];
  relatedFactionNames?: string[];
  confidenceScore?: number;
  confidenceLabel?: "high" | "medium" | "low";
  needsReview?: boolean;
}
