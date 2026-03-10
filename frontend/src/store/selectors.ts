import type { CampaignContextState } from "./campaignContext";
import type { Campaign, Session, Scene, Npc, Voice, CodexEntry, ActionLogEvent, NarrationClip } from "../types";

/** Pure selectors — take state and return derived data. */

export function getActiveCampaign(state: CampaignContextState): Campaign | null {
  if (!state.activeCampaignId) return null;
  return state.campaigns.find((c) => c.id === state.activeCampaignId) ?? null;
}

export function getActiveSession(state: CampaignContextState): Session | null {
  if (!state.activeSessionId) return null;
  return state.sessions.find((s) => s.id === state.activeSessionId) ?? null;
}

export function getActiveScene(state: CampaignContextState): Scene | null {
  if (!state.activeSceneId) return null;
  return state.scenes.find((s) => s.id === state.activeSceneId) ?? null;
}

export function getSceneNpcs(state: CampaignContextState, sceneId: string): Npc[] {
  const scene = state.scenes.find((s) => s.id === sceneId);
  if (!scene?.npcIds?.length) return [];
  const ids = new Set(scene.npcIds);
  return state.npcs.filter((n) => ids.has(n.id));
}

export function getSceneCodexEntries(state: CampaignContextState, sceneId: string): CodexEntry[] {
  const scene = state.scenes.find((s) => s.id === sceneId);
  if (!scene?.codexEntryIds?.length) return [];
  const ids = new Set(scene.codexEntryIds);
  return state.codexEntries.filter((e) => ids.has(e.id));
}

export function getNpcVoice(state: CampaignContextState, npcId: string): Voice | null {
  const npc = state.npcs.find((n) => n.id === npcId);
  if (!npc?.voiceId) return null;
  return state.voices.find((v) => v.id === npc.voiceId) ?? null;
}

/** Returns the single voice assigned to this NPC, if any (as array for consistency). */
export function getVoicesForNpc(state: CampaignContextState, npcId: string): Voice[] {
  const voice = getNpcVoice(state, npcId);
  return voice ? [voice] : [];
}

export function getActionLogForActiveScene(state: CampaignContextState): ActionLogEvent[] {
  const sceneId = state.activeSceneId;
  if (!sceneId) return state.actionLog;

  const scene = state.scenes.find((s) => s.id === sceneId);
  if (!scene) return [];

  const ids = scene.actionLogIds;
  if (!ids?.length) {
    return state.actionLog.filter((e) => e.sceneId === sceneId);
  }

  const idSet = new Set(ids);
  return state.actionLog.filter((e) => idSet.has(e.id));
}

export function getNarrationClipsForActiveScene(state: CampaignContextState): NarrationClip[] {
  const sceneId = state.activeSceneId;
  if (!sceneId) return [];
  const scene = state.scenes.find((s) => s.id === sceneId);
  if (!scene) return [];

  const ids = scene.narrationClipIds;
  if (!ids?.length) {
    return state.narrationClips.filter((c) => c.sceneId === sceneId);
  }

  const idSet = new Set(ids);
  return state.narrationClips.filter((c) => idSet.has(c.id));
}

export function getRelatedCodexForNpc(state: CampaignContextState, npcId: string): CodexEntry[] {
  return state.codexEntries.filter(
    (e) => e.relatedNpcIds?.includes(npcId)
  );
}

export function getCodexEntriesForCampaign(state: CampaignContextState): CodexEntry[] {
  const cid = state.activeCampaignId;
  if (!cid) return state.codexEntries;
  return state.codexEntries.filter((e) => e.campaignId === cid);
}

export function getNpcsForActiveCampaign(state: CampaignContextState): Npc[] {
  const cid = state.activeCampaignId;
  if (!cid) return state.npcs;
  return state.npcs.filter((n) => n.campaignId === cid);
}

/** NPCs that have the given voice assigned. */
export function getNpcsForVoice(state: CampaignContextState, voiceId: string): Npc[] {
  const voice = state.voices.find((v) => v.id === voiceId);
  if (!voice?.assignedNpcIds?.length) return [];
  const ids = new Set(voice.assignedNpcIds);
  return state.npcs.filter((n) => ids.has(n.id));
}

/** Hook wrappers — subscribe to store and return selector results. */
import { useCampaignContextStore } from "./campaignContext";

function useStoreSelector<T>(selector: (state: CampaignContextState) => T): T {
  return useCampaignContextStore(selector);
}

export function useActiveCampaign(): Campaign | null {
  return useStoreSelector(getActiveCampaign);
}

export function useActiveSession(): Session | null {
  return useStoreSelector(getActiveSession);
}

export function useActiveScene(): Scene | null {
  return useStoreSelector(getActiveScene);
}

export function useSceneNpcs(sceneId: string | null | undefined): Npc[] {
  const state = useStoreSelector((s) => s);
  const sid = sceneId ?? state.activeSceneId;
  return sid ? getSceneNpcs(state, sid) : [];
}

export function useSceneCodexEntries(sceneId: string | null | undefined): CodexEntry[] {
  const state = useStoreSelector((s) => s);
  const sid = sceneId ?? state.activeSceneId;
  return sid ? getSceneCodexEntries(state, sid) : [];
}

export function useVoices(): Voice[] {
  return useStoreSelector((s) => s.voices);
}

export function useNpcVoice(npcId: string | null | undefined): Voice | null {
  const state = useStoreSelector((s) => s);
  return npcId ? getNpcVoice(state, npcId) : null;
}

export function useVoicesForNpc(npcId: string | null | undefined): Voice[] {
  const state = useStoreSelector((s) => s);
  return npcId ? getVoicesForNpc(state, npcId) : [];
}

export function useActionLogForActiveScene(): ActionLogEvent[] {
  return useStoreSelector(getActionLogForActiveScene);
}

export function useNarrationClipsForActiveScene(): NarrationClip[] {
  return useStoreSelector(getNarrationClipsForActiveScene);
}

export function useCodexEntriesForActiveCampaign(): CodexEntry[] {
  return useStoreSelector(getCodexEntriesForCampaign);
}

export function useNpcsForActiveCampaign(): Npc[] {
  return useStoreSelector(getNpcsForActiveCampaign);
}

export function useNpcsForVoice(voiceId: string | null | undefined): Npc[] {
  const state = useStoreSelector((s) => s);
  return voiceId ? getNpcsForVoice(state, voiceId) : [];
}
