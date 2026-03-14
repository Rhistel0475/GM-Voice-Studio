/**
 * Player Session Portal — aggregate data for player-facing view: recaps, timeline, locations, NPCs, lore.
 */

import type { Campaign, Session, Scene, Npc, CodexEntry, ActionLogEvent } from "../types";
import { buildCampaignTimeline } from "./timeline";
import type { CampaignTimeline } from "./timeline";
import type { PlayerRecap } from "./recaps";

export interface PlayerPortalData {
  campaign: Campaign | null;
  sessions: Session[];
  sessionRecaps: PlayerRecap[];
  timeline: CampaignTimeline;
  discoveredLocations: { id: string; name: string; summary?: string }[];
  knownNpcs: { id: string; name: string; role?: string; summary?: string }[];
  importantLore: { id: string; title: string; summary?: string; type?: string }[];
}

export interface GeneratePlayerPortalOptions {
  campaign: Campaign | null;
  sessions: Session[];
  scenes: Scene[];
  npcs: Npc[];
  codexEntries: CodexEntry[];
  actionLog: ActionLogEvent[];
  sessionRecaps?: PlayerRecap[];
}

/**
 * Generate player-facing portal data: recaps, campaign timeline, discovered locations, known NPCs, important lore.
 */
export function generatePlayerPortalData(options: GeneratePlayerPortalOptions): PlayerPortalData {
  const {
    campaign,
    sessions,
    scenes,
    npcs,
    codexEntries,
    actionLog,
    sessionRecaps = [],
  } = options;

  const campaignId = campaign?.id ?? null;
  const timeline = buildCampaignTimeline(
    campaign ?? { id: "", name: "", activeSessionId: undefined },
    sessions,
    scenes,
    actionLog
  );

  const discoveredLocations = Array.from(
    new Map(
      scenes
        .filter((s) => s.campaignId === campaignId)
        .map((s) => [
          s.locationId ?? s.id,
          {
            id: s.locationId ?? s.id,
            name: s.title,
            summary: s.summary,
          },
        ])
    ).values()
  );

  const knownNpcIds = new Set(
    scenes.flatMap((s) => (s.campaignId === campaignId ? s.npcIds ?? [] : []))
  );
  const knownNpcs = npcs
    .filter((n) => n.campaignId === campaignId && knownNpcIds.has(n.id))
    .map((n) => ({
      id: n.id,
      name: n.name,
      role: n.role,
      summary: n.summary,
    }));

  const importantLore = codexEntries
    .filter((e) => e.campaignId === campaignId && ["lore", "rule", "document"].includes(e.type))
    .slice(0, 50)
    .map((e) => ({
      id: e.id,
      title: e.title,
      summary: e.summary,
      type: e.type,
    }));

  return {
    campaign,
    sessions,
    sessionRecaps,
    timeline,
    discoveredLocations,
    knownNpcs,
    importantLore,
  };
}
