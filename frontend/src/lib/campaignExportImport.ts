/**
 * Campaign export / import — NPCs, scenes, codex, encounters, timeline, memories.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import { useCampaignMemoryStore } from "./campaignMemory";
import type {
  Campaign,
  Session,
  Scene,
  Npc,
  CodexEntry,
  ActionLogEvent,
  NarrationClip,
} from "../types";
import type { CampaignMemoryEvent } from "./campaignMemory";
import { buildCampaignTimeline } from "./timeline";

export interface CampaignExportPayload {
  version: number;
  exportedAt: string;
  campaign: Campaign;
  sessions: Session[];
  scenes: Scene[];
  npcs: Npc[];
  codexEntries: CodexEntry[];
  actionLog: ActionLogEvent[];
  narrationClips: NarrationClip[];
  memories: CampaignMemoryEvent[];
  timeline?: { campaignId: string; events: unknown[] };
}

/**
 * Export a campaign to a JSON blob (NPCs, scenes, codex, action log, clips, memories, timeline).
 */
export function exportCampaign(campaignId: string): CampaignExportPayload {
  const state = useCampaignContextStore.getState();
  const memoryState = useCampaignMemoryStore.getState();

  const campaign = state.campaigns.find((c) => c.id === campaignId) ?? null;
  if (!campaign) {
    throw new Error(`Campaign not found: ${campaignId}`);
  }

  const sessions = state.sessions.filter((s) => s.campaignId === campaignId);
  const scenes = state.scenes.filter((s) => s.campaignId === campaignId);
  const npcs = state.npcs.filter((n) => n.campaignId === campaignId);
  const codexEntries = state.codexEntries.filter((e) => e.campaignId === campaignId);
  const sessionIds = new Set(sessions.map((s) => s.id));
  const actionLog = state.actionLog.filter(
    (e) => !e.sessionId || sessionIds.has(e.sessionId)
  );
  const narrationClips = state.narrationClips.filter((c) => c.campaignId === campaignId);
  const memories = memoryState.memories.filter(
    (m) => m.sessionId && sessionIds.has(m.sessionId)
  );

  const timeline = buildCampaignTimeline(campaign, sessions, scenes, actionLog);

  return {
    version: 1,
    exportedAt: new Date().toISOString(),
    campaign,
    sessions,
    scenes,
    npcs,
    codexEntries,
    actionLog,
    narrationClips,
    memories,
    timeline: { campaignId: timeline.campaignId, events: timeline.events },
  };
}

/**
 * Import a campaign from a JSON file/blob. Merges into the store (adds campaigns, sessions, scenes, npcs, codex, log, clips, memories).
 */
export function importCampaign(payload: CampaignExportPayload): Campaign {
  const store = useCampaignContextStore.getState();
  const memoryStore = useCampaignMemoryStore.getState();

  const {
    campaign,
    sessions,
    scenes,
    npcs,
    codexEntries,
    actionLog,
    narrationClips,
    memories,
  } = payload;

  const existingIds = {
    campaigns: new Set(store.campaigns.map((c) => c.id)),
    sessions: new Set(store.sessions.map((s) => s.id)),
    scenes: new Set(store.scenes.map((s) => s.id)),
    npcs: new Set(store.npcs.map((n) => n.id)),
    codexEntries: new Set(store.codexEntries.map((e) => e.id)),
  };

  const campaignExists = existingIds.campaigns.has(campaign.id);
  const nextCampaigns = campaignExists
    ? store.campaigns.map((c) => (c.id === campaign.id ? campaign : c))
    : [...store.campaigns, campaign];

  const sessionsToAdd = sessions.filter((s) => !existingIds.sessions.has(s.id));
  const scenesToAdd = scenes.filter((s) => !existingIds.scenes.has(s.id));
  const npcsToAdd = npcs.filter((n) => !existingIds.npcs.has(n.id));
  const codexToAdd = codexEntries.filter((e) => !existingIds.codexEntries.has(e.id));

  useCampaignContextStore.setState({
    campaigns: nextCampaigns,
    sessions: [...store.sessions, ...sessionsToAdd],
    scenes: [...store.scenes, ...scenesToAdd],
    npcs: [...store.npcs, ...npcsToAdd],
    codexEntries: [...store.codexEntries, ...codexToAdd],
    actionLog: [...store.actionLog, ...actionLog],
    narrationClips: [...store.narrationClips, ...narrationClips],
    activeCampaignId: campaign.id,
    activeSessionId: sessions[0]?.id ?? null,
    activeSceneId: sessions[0]?.activeSceneId ?? scenes[0]?.id ?? null,
  });

  for (const m of memories) {
    memoryStore.recordCampaignMemory({
      category: m.category,
      summary: m.summary,
      details: m.details,
      npcId: m.npcId,
      factionId: m.factionId,
      questId: m.questId,
      sessionId: m.sessionId,
      sceneId: m.sceneId,
    });
  }

  return campaign;
}
