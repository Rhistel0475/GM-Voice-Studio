import React, { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { useAppState } from "./AppStateContext";
import { getNpcsForScene } from "../types/campaign";

const ACTION_LOG_MAX = 100;

const CampaignContext = createContext(null);

/**
 * Campaign context value shape.
 * @typedef {Object} CampaignContextValue
 * @property {import("../types/campaign").Campaign|null} campaign
 * @property {Function} setCampaign
 * @property {number} activeSceneIndex
 * @property {Function} setActiveSceneIndex
 * @property {{ sessionStartMs: number|null }} activeSession
 * @property {import("../types/campaign").ActionLogEntry[]} actionLog
 * @property {Function} appendActionLog
 * @property {import("../types/campaign").NarrationClip[]} narrationClips
 * @property {Function} addNarrationClip
 * @property {Array} npcs
 * @property {Array} voices
 * @property {Function} assignVoiceToNpc
 * @property {Function} addCodexEntryToScene
 * @property {Function} assignNpcToScene
 */

export function CampaignProvider({ children }) {
  const { campaignData, setCampaignData } = useAppState();

  const [activeSceneIndex, setActiveSceneIndexState] = useState(() => {
    try {
      const s = sessionStorage.getItem("gm_active_scene_index");
      return s != null ? Math.max(0, parseInt(s, 10)) : 0;
    } catch {
      return 0;
    }
  });
  const [actionLog, setActionLog] = useState([]);
  const [narrationClips, setNarrationClips] = useState([]);
  const [voices, setVoices] = useState([]);
  const sessionStartMsRef = useRef(typeof Date.now === "function" ? Date.now() : null);

  const setActiveSceneIndex = useCallback((idx) => {
    setActiveSceneIndexState(idx);
    try {
      sessionStorage.setItem("gm_active_scene_index", String(idx));
    } catch {
      /* ignore */
    }
  }, []);

  const appendActionLog = useCallback((role, text, meta = "") => {
    if (!text) return;
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      role,
      text,
      meta: meta || undefined,
    };
    setActionLog((prev) => [...prev, entry].slice(-ACTION_LOG_MAX));
    // TODO: Backend — POST /api/session/events or send via websocket for live sync.
  }, []);

  const addNarrationClip = useCallback((clip) => {
    const id = clip?.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setNarrationClips((prev) => [...prev, { ...clip, id, createdAt: clip?.createdAt || new Date().toISOString() }]);
    // TODO: Backend — persist narration clips per session/campaign if needed.
  }, []);

  const assignVoiceToNpc = useCallback(
    (npcIdOrName, voiceId) => {
      const campaign = campaignData;
      if (!campaign?.npcs?.length) return;
      const nameOrId = String(npcIdOrName);
      const updated = campaign.npcs.map((n) =>
        (n.name === nameOrId || n.id === nameOrId) ? { ...n, voice_id: voiceId, voiceId } : n
      );
      setCampaignData({ ...campaign, npcs: updated });
      // TODO: Backend — PATCH /api/campaigns/:id/npcs/:npcId or PATCH /api/npcs/:id to persist voice assignment.
    },
    [campaignData, setCampaignData]
  );

  const addCodexEntryToScene = useCallback(
    (codexEntryIdOrRef, sceneIndex) => {
      const campaign = campaignData;
      if (!campaign?.scenes?.length) return;
      const idx = sceneIndex ?? activeSceneIndex;
      const scene = campaign.scenes[idx];
      if (!scene) return;
      const codexRefs = Array.isArray(scene.codexRefs) ? [...scene.codexRefs] : [];
      if (codexRefs.includes(codexEntryIdOrRef)) return;
      codexRefs.push(String(codexEntryIdOrRef));
      const updated = campaign.scenes.map((s, i) => (i === idx ? { ...s, codexRefs } : s));
      setCampaignData({ ...campaign, scenes: updated });
      // TODO: Backend — PATCH campaign scenes / codex refs.
    },
    [campaignData, setCampaignData, activeSceneIndex]
  );

  const assignNpcToScene = useCallback(
    (npcNameOrId, sceneIndex) => {
      const campaign = campaignData;
      if (!campaign?.scenes?.length) return;
      const idx = sceneIndex ?? activeSceneIndex;
      const scene = campaign.scenes[idx];
      if (!scene) return;
      const npcNames = Array.isArray(scene.npcs) ? [...scene.npcs] : [];
      const name = String(npcNameOrId);
      if (npcNames.includes(name)) return;
      npcNames.push(name);
      const updated = campaign.scenes.map((s, i) => (i === idx ? { ...s, npcs: npcNames } : s));
      setCampaignData({ ...campaign, scenes: updated });
      // TODO: Backend — PATCH campaign scenes to persist NPC assignment.
    },
    [campaignData, setCampaignData, activeSceneIndex]
  );

  const activeSession = useMemo(
    () => ({ sessionStartMs: sessionStartMsRef.current, activeSceneIndex }),
    [activeSceneIndex]
  );

  const npcs = campaignData?.npcs ?? [];
  const activeScene = campaignData?.scenes?.[activeSceneIndex] ?? null;

  const value = useMemo(
    () => ({
      campaign: campaignData,
      setCampaign: setCampaignData,
      activeSceneIndex,
      setActiveSceneIndex,
      activeSession,
      activeScene,
      actionLog,
      appendActionLog,
      narrationClips,
      addNarrationClip,
      npcs,
      voices,
      setVoices,
      assignVoiceToNpc,
      addCodexEntryToScene,
      assignNpcToScene,
    }),
    [
      campaignData,
      setCampaignData,
      activeSceneIndex,
      setActiveSceneIndex,
      activeSession,
      activeScene,
      actionLog,
      appendActionLog,
      narrationClips,
      addNarrationClip,
      npcs,
      voices,
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
 * @param {import("../types/campaign").Scene|null} [scene] - If omitted, uses active scene from context.
 */
export function useNpcsForScene(scene) {
  const { campaign, activeScene } = useCampaign();
  const targetScene = scene ?? activeScene;
  return useMemo(() => getNpcsForScene(campaign, targetScene), [campaign, targetScene]);
}
