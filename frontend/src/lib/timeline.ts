import type { Campaign, Session, Scene, ActionLogEvent } from "../types";

export interface TimelineEvent {
  id: string;
  kind: "session" | "scene" | "event" | "faction" | "npc";
  label: string;
  description?: string;
  timestamp?: string;
  sessionId?: string;
  sceneId?: string;
  npcId?: string;
  factionId?: string;
}

export interface CampaignTimeline {
  campaignId: string;
  events: TimelineEvent[];
}

export function buildCampaignTimeline(
  campaign: Campaign,
  sessions: Session[],
  scenes: Scene[],
  log: ActionLogEvent[]
): CampaignTimeline {
  const events: TimelineEvent[] = [];

  sessions
    .filter((s) => s.campaignId === campaign.id)
    .forEach((session) => {
      events.push({
        id: `session-${session.id}`,
        kind: "session",
        label: session.title,
        description: "Play session",
        timestamp: session.startedAt,
        sessionId: session.id,
      });
    });

  scenes
    .filter((scene) => scene.campaignId === campaign.id)
    .forEach((scene) => {
      events.push({
        id: `scene-${scene.id}`,
        kind: "scene",
        label: scene.title,
        description: scene.summary,
        sceneId: scene.id,
      });
    });

  log
    .filter((e) => !!e.createdAt)
    .forEach((e) => {
      events.push({
        id: `event-${e.id}`,
        kind: "event",
        label: e.text.slice(0, 80),
        description: e.text,
        timestamp: e.createdAt,
        sessionId: e.sessionId,
        sceneId: e.sceneId,
      });
    });

  events.sort((a, b) => {
    if (!a.timestamp && !b.timestamp) return 0;
    if (!a.timestamp) return 1;
    if (!b.timestamp) return -1;
    return a.timestamp.localeCompare(b.timestamp);
  });

  return { campaignId: campaign.id, events };
}

