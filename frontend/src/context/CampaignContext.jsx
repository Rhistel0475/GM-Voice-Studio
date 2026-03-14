import { createContext, useCallback, useContext, useEffect, useMemo } from "react";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  getActiveCampaign,
  getActiveSession,
  getActiveScene,
  getActionLogForActiveScene,
  getNarrationClipsForActiveScene,
} from "../store/selectors";
import { getNpcsForScene } from "../types/campaign";
import { importParseResultToStore } from "../lib/campaignImport";

const CampaignContext = createContext(null);

/** Map store action log type to legacy role (player | assistant | stat_block | lore | error). */
function eventTypeToRole(type) {
  const t = type?.toLowerCase?.();
  if (t === "player") return "player";
  if (t === "npc" || t === "narration") return "assistant";
  if (t === "system") return "stat_block";
  return "assistant";
}

/** Map legacy role to store action log type. */
function roleToEventType(role) {
  const r = role?.toLowerCase?.();
  if (r === "player") return "player";
  if (r === "assistant") return "npc";
  if (r === "stat_block" || r === "lore") return "system";
  if (r === "error") return "system";
  return "npc";
}

/** Build legacy campaign shape from store state for existing components. */
function deriveLegacyCampaign(state) {
  const campaign = getActiveCampaign(state);
  if (!campaign) return null;
  const campaignScenes = state.scenes.filter((s) => s.campaignId === campaign.id);
  const campaignNpcs = state.npcs.filter((n) => n.campaignId === campaign.id);
  return {
    id: campaign.id,
    title: campaign.name,
    scenes: campaignScenes.map((s) => ({
      id: s.id,
      title: s.title,
      summary: s.summary,
      npcs: s.npcIds
        .map((id) => campaignNpcs.find((n) => n.id === id)?.name)
        .filter(Boolean),
      codexRefs: s.codexEntryIds ?? [],
      items: [],
      reveals: [],
    })),
    npcs: campaignNpcs.map((n) => ({
      id: n.id,
      name: n.name,
      role: n.role,
      personality: n.summary,
      voice_id: n.voiceId,
      voiceId: n.voiceId,
    })),
    party: [],
    items: [],
    reveals: [],
    locations: [],
  };
}

/** Build legacy activeScene from store (object with title, npcs names, codexRefs). */
function deriveLegacyActiveScene(state) {
  const scene = getActiveScene(state);
  if (!scene) return null;
  const campaign = getActiveCampaign(state);
  const campaignNpcs = campaign ? state.npcs.filter((n) => n.campaignId === campaign.id) : state.npcs;
  return {
    id: scene.id,
    title: scene.title,
    summary: scene.summary,
    npcs: scene.npcIds.map((id) => campaignNpcs.find((n) => n.id === id)?.name).filter(Boolean),
    codexRefs: scene.codexEntryIds ?? [],
    items: [],
    reveals: [],
  };
}

/** Build legacy action log entries (id, role, text, meta). */
function deriveLegacyActionLog(state) {
  const events = getActionLogForActiveScene(state);
  return events.map((e) => ({
    id: e.id,
    role: eventTypeToRole(e.type),
    text: e.text,
    meta: e.createdAt ?? "",
  }));
}

export function CampaignProvider({ children }) {
  const state = useCampaignContextStore((s) => s);

  useEffect(() => {
    if (state.activeCampaignId || state.campaigns.length > 0) return;
    // Import previously-saved parse result if present; otherwise stay empty.
    try {
      const saved = localStorage.getItem("gm_campaign_data");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed) && parsed.title) {
          importParseResultToStore(parsed);
        }
      }
    } catch {
      // localStorage unreadable or corrupt — stay empty.
    }
  }, [state.activeCampaignId, state.campaigns.length]);

  const campaignScenes = useMemo(
    () => (state.activeCampaignId ? state.scenes.filter((s) => s.campaignId === state.activeCampaignId) : []),
    [state.activeCampaignId, state.scenes]
  );
  const activeSceneIndex = useMemo(
    () => (state.activeSceneId ? campaignScenes.findIndex((s) => s.id === state.activeSceneId) : 0),
    [state.activeSceneId, campaignScenes]
  );

  const setActiveSceneIndex = useCallback(
    (idx) => {
      const scene = campaignScenes[idx];
      if (scene) state.setActiveScene(scene.id);
    },
    [campaignScenes, state.setActiveScene]
  );

  const appendActionLog = useCallback(
    (role, text, _meta = "") => {
      if (!text) return;
      state.addActionLogEvent({ type: roleToEventType(role), text });
    },
    [state.addActionLogEvent]
  );

  const addNarrationClip = useCallback(
    (clip) => {
      state.addNarrationClip({
        title: clip?.title ?? "Untitled",
        voiceId: clip?.voiceId,
        audioUrl: clip?.audioUrl,
        duration: clip?.duration,
      });
    },
    [state.addNarrationClip]
  );

  const assignVoiceToNpc = useCallback(
    (npcIdOrName, voiceId) => {
      const nameOrId = String(npcIdOrName);
      const npc = state.npcs.find((n) => n.id === nameOrId || n.name === nameOrId);
      if (npc) state.assignVoiceToNpc(npc.id, voiceId);
    },
    [state.npcs, state.assignVoiceToNpc]
  );

  const addCodexEntryToScene = useCallback(
    (codexEntryIdOrRef, sceneIndex) => {
      const idx = sceneIndex ?? activeSceneIndex;
      const scene = campaignScenes[idx];
      if (scene) state.addCodexEntryToScene(scene.id, String(codexEntryIdOrRef));
    },
    [campaignScenes, activeSceneIndex, state.addCodexEntryToScene]
  );

  const assignNpcToScene = useCallback(
    (npcNameOrId, sceneIndex) => {
      const nameOrId = String(npcNameOrId);
      const npc = state.npcs.find((n) => n.id === nameOrId || n.name === nameOrId);
      if (!npc) return;
      const idx = sceneIndex ?? activeSceneIndex;
      const scene = campaignScenes[idx];
      if (scene) state.assignNpcToScene(scene.id, npc.id);
    },
    [state.npcs, campaignScenes, activeSceneIndex, state.assignNpcToScene]
  );

  const sessionStartMs = useMemo(() => {
    const session = getActiveSession(state);
    if (session?.startedAt) return new Date(session.startedAt).getTime();
    return null;
  }, [state]);

  const activeSession = useMemo(
    () => ({ sessionStartMs, activeSceneIndex }),
    [sessionStartMs, activeSceneIndex]
  );

  const campaign = useMemo(() => deriveLegacyCampaign(state), [state]);
  const activeScene = useMemo(() => deriveLegacyActiveScene(state), [state]);
  const actionLog = useMemo(() => deriveLegacyActionLog(state), [state]);
  const narrationClips = useMemo(() => getNarrationClipsForActiveScene(state), [state]);

  const value = useMemo(
    () => ({
      campaign,
      setCampaign: () => {}, // No-op for now; store is source of truth
      activeSceneIndex: Math.max(0, activeSceneIndex),
      setActiveSceneIndex,
      activeSession,
      activeScene,
      actionLog,
      appendActionLog,
      narrationClips,
      addNarrationClip,
      npcs: state.npcs,
      voices: state.voices,
      setVoices: state.setVoices,
      assignVoiceToNpc,
      addCodexEntryToScene,
      assignNpcToScene,
    }),
    [
      campaign,
      activeSceneIndex,
      setActiveSceneIndex,
      activeSession,
      activeScene,
      actionLog,
      appendActionLog,
      narrationClips,
      addNarrationClip,
      state.npcs,
      state.voices,
      state.setVoices,
      assignVoiceToNpc,
      addCodexEntryToScene,
      assignNpcToScene,
    ]
  );

  return <CampaignContext.Provider value={value}>{children}</CampaignContext.Provider>;
}

export function useCampaign() {
  const ctx = useContext(CampaignContext);
  if (!ctx) throw new Error("useCampaign must be used within CampaignProvider");
  return ctx;
}

/** Returns context value or null when used outside CampaignProvider (for gradual adoption). */
export function useCampaignOptional() {
  return useContext(CampaignContext);
}

export function useActiveCampaign() {
  return useCampaign().campaign;
}

export function useActiveSession() {
  const { activeSession } = useCampaign();
  return activeSession;
}

export function useActiveScene() {
  return useCampaign().activeScene;
}

export function useActionLog() {
  return useCampaign().actionLog;
}

/**
 * Returns NPCs that appear in the given scene (or the active scene if scene is omitted).
 * Uses legacy campaign shape for compatibility; when store is used, resolves via getSceneNpcs.
 */
export function useNpcsForScene(scene) {
  const ctx = useCampaign();
  const { campaign, activeScene } = ctx;
  const targetScene = scene ?? activeScene;
  return useMemo(() => getNpcsForScene(campaign, targetScene), [campaign, targetScene]);
}
