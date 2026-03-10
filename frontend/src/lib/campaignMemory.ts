/**
 * Campaign Memory System — long-term memory for NPC interactions, quests, faction changes, player decisions.
 * Influences future AI responses.
 */

import { create } from "zustand";
import { createId, nowIso } from "./utils/ids";

export type CampaignMemoryCategory =
  | "npc_interaction"
  | "quest_resolved"
  | "faction_change"
  | "player_decision"
  | "major_event";

export interface CampaignMemoryEvent {
  id: string;
  category: CampaignMemoryCategory;
  summary: string;
  details?: string;
  npcId?: string;
  factionId?: string;
  questId?: string;
  sessionId?: string;
  sceneId?: string;
  createdAt: string;
}

interface CampaignMemoryState {
  memories: CampaignMemoryEvent[];
}

const initialState: CampaignMemoryState = {
  memories: [],
};

export const useCampaignMemoryStore = create<CampaignMemoryState & {
  recordCampaignMemory: (event: Omit<CampaignMemoryEvent, "id" | "createdAt">) => void;
  retrieveRelevantMemories: (context: {
    npcIds?: string[];
    factionIds?: string[];
    sessionId?: string;
    sceneId?: string;
    limit?: number;
  }) => CampaignMemoryEvent[];
  clearMemories: () => void;
}>((set, get) => ({
  ...initialState,

  recordCampaignMemory(event) {
    const memory: CampaignMemoryEvent = {
      ...event,
      id: createId("memory"),
      createdAt: nowIso(),
    };
    set((state) => ({
      memories: [...state.memories, memory].slice(-500),
    }));
  },

  retrieveRelevantMemories(context) {
    const { memories } = get();
    const limit = context.limit ?? 20;
    const npcSet = context.npcIds?.length ? new Set(context.npcIds) : null;
    const factionSet = context.factionIds?.length ? new Set(context.factionIds) : null;

    const scored = memories.map((m) => {
      let score = 0;
      if (context.sessionId && m.sessionId === context.sessionId) score += 3;
      if (context.sceneId && m.sceneId === context.sceneId) score += 2;
      if (npcSet && m.npcId && npcSet.has(m.npcId)) score += 2;
      if (factionSet && m.factionId && factionSet.has(m.factionId)) score += 2;
      if (!npcSet && !factionSet && !context.sessionId) score = 1;
      return { memory: m, score };
    });

    return scored
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((s) => s.memory);
  },

  clearMemories() {
    set({ memories: [] });
  },
}));

/** Record a campaign memory (convenience). */
export function recordCampaignMemory(
  event: Omit<CampaignMemoryEvent, "id" | "createdAt">
): void {
  useCampaignMemoryStore.getState().recordCampaignMemory(event);
}

/** Retrieve memories relevant to the given context (convenience). */
export function retrieveRelevantMemories(context: {
  npcIds?: string[];
  factionIds?: string[];
  sessionId?: string;
  sceneId?: string;
  limit?: number;
}): CampaignMemoryEvent[] {
  return useCampaignMemoryStore.getState().retrieveRelevantMemories(context);
}
