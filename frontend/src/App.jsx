import { useCallback, useEffect, useMemo, useRef, useState, Component } from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import { AppStateProvider, useAppState } from "./context/AppStateContext";
import { CampaignProvider, useCampaignOptional } from "./context/CampaignContext";
import { useCampaignContextStore } from "./store/campaignContext";
import { useExtractionReviewQueueStore } from "./store/extractionReview";
import { parseResultToExtractionBatch } from "./lib/parseResultToExtractionBatch";
import { importParseResultToStore } from "./lib/campaignImport";
import { getBackendCampaignId, setBackendCampaignId, persistSessionEvent } from "./lib/campaignPersistence";
import {
  DEFAULT_GAME_SYSTEM_ID,
  listGameSystemPlugins,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
  resolveGameSystemPlugin,
} from "./lib/gameSystemPlugins";
import AppShell from "./layout/AppShell";
import PrepRoom from "./components/prep/PrepRoom";
import AdventureIntake from "./components/prep/AdventureIntake";
import CommandPalette from "./components/layout/CommandPalette";
import LiveBoardPage from "./app/live-board";
import CodexPage from "./app/codex";
import NPCWorkshopPage from "./app/npcs";
import SettingsPage from "./pages/SettingsPage";
import PrepPage from "./pages/PrepPage";
import VoicePage from "./pages/VoicePage";
import CampaignPage from "./pages/CampaignPage";
import NpcVoiceModal from "./components/live-board/NpcVoiceModal";
import { addSessionLogEntry } from "./lib/liveboardCampaignContext";
import { buildSessionAssistantContext, buildSessionAssistantSuggestions } from "./lib/sessionAssistant";

class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(err) { return { error: err }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{ background: "#1a0f06", color: "#ff6b6b", padding: "2rem", fontFamily: "monospace", fontSize: "13px", minHeight: "100vh" }}>
          <div style={{ color: "#d4af37", fontFamily: "Cinzel,serif", fontSize: "1.1rem", marginBottom: "1rem" }}>Library Render Error</div>
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#ff6b6b" }}>{this.state.error?.message}</pre>
          <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#9b7440", marginTop: "0.5rem", fontSize: "11px" }}>{this.state.error?.stack}</pre>
          <button
            style={{ marginTop: "1.5rem", background: "#2a1a0a", border: "1px solid #c8a050", color: "#c8a050", padding: "0.5rem 1rem", cursor: "pointer", borderRadius: "4px" }}
            onClick={() => {
              localStorage.removeItem("gm_parse_result");
              localStorage.removeItem("gm_parse_images");
              localStorage.removeItem("gm_campaign_data");
              setBackendCampaignId(null);
              useCampaignContextStore.getState().resetCampaignContext();
              useExtractionReviewQueueStore.getState().clearQueue();
              window.location.reload();
            }}
          >Clear saved data &amp; reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}
import {
  BookOpenText,
  LayoutDashboard,
  Map,
  Mic2,
  ScrollText,
  Sparkles,
  Upload,
  Volume2,
  Zap,
  Save,
  CheckCircle,
  Trash2,
} from "lucide-react";

const pathToView = (path) => {
  let p = (path || "/").replace(/\/+$/, "").trim() || "/";
  if (!p.startsWith("/")) p = `/${p}`;
  if (p === "/codex") return "codex";
  if (p === "/npcs") return "npc-workshop";
  if (p === "/voices") return "voice-studio";
  if (p === "/settings") return "settings";
  if (p === "/prep") return "prep";
  if (p === "/intake") return "intake";
  if (p === "/campaign") return "campaign";
  return "live";
};

const viewToPath = {
  live: "/",
  codex: "/codex",
  "npc-workshop": "/npcs",
  "voice-studio": "/voices",
  settings: "/settings",
  prep: "/prep",
  intake: "/prep?mode=upload", // remapped: Library now opens PrepPage in upload mode
  campaign: "/campaign",
};
const buildWebSocketUrl = (path) => {
  if (typeof window === "undefined") return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
};
const normalizeWakeText = (text) => (text || "")
  .toLowerCase()
  .replace(/[^a-z0-9\s]/g, " ")
  .replace(/\s+/g, " ")
  .trim();
const WAKE_WORD = normalizeWakeText(import.meta.env.VITE_WAKE_WORD || "hey co gm");
const SILENCE_RMS_THRESHOLD = 0.015;
const SILENCE_HOLD_MS = 2200;

// Fallback defaults (shown when no campaign data is loaded)
const DEFAULT_SCENES = [
  { title: "No scenes loaded", act: "Upload docs in Library", type: "exploration", atmosphere_type: "forest", read_aloud: "", npcs: [], location: "", notes: "" },
];

const resolvePreferredNpc = (campaign, activeScene, selectedNpcName = null) => {
  const allNpcs = campaign?.npcs || [];
  if (!allNpcs.length) return null;

  if (selectedNpcName) {
    const selectedNpc = allNpcs.find((npc) => npc.name === selectedNpcName || npc.id === selectedNpcName);
    if (selectedNpc) return selectedNpc;
  }

  const sceneNpcNames = activeScene?.npcs || [];
  const sceneNpc = sceneNpcNames
    .map((npcName) => allNpcs.find((npc) => npc.name === npcName || npc.id === npcName))
    .find(Boolean);

  return sceneNpc || allNpcs[0] || null;
};

const resolveEncounterRef = (scene) => String(
  scene?.encounter_id
  || scene?.encounterId
  || scene?.id
  || scene?.title
  || ""
).trim();

// ─── Live Board ─────────────────────────────────────────────────────────────

const formatSessionTimer = (startMs) => {
  if (!startMs) return "0:00";
  const elapsed = Math.max(0, Math.floor((Date.now() - startMs) / 1000));
  const m = Math.floor(elapsed / 60);
  const s = elapsed % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
};
const LiveBoard = ({
  view: _view,
  onNavigate,
  campaignData,
  authFetch,
  setBannerState,
  defaultAutoQueryOnVoice = true,
  onRegisterCommandActions,
  onSessionStarted,
}) => {
  const campaignCtx = useCampaignOptional();
  const useSharedCampaign = Boolean(campaignCtx);

  const [selectedSceneIdxLocal, setSelectedSceneIdxLocal] = useState(0);
  const [selectedNpcName, setSelectedNpcName] = useState(null);
  const [actionLogLocal, setActionLogLocal] = useState([]);
  const [coDmQuery, setCoDmQuery] = useState("");
  const [isSubmittingQuery, setIsSubmittingQuery] = useState(false);
  const [coDmStatus, setCoDmStatus] = useState("connecting");
  const [isMicActive, setIsMicActive] = useState(false);
  const [micError, setMicError] = useState("");
  const [isWakeArmed, setIsWakeArmed] = useState(false);
  const [wakeError, setWakeError] = useState("");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [autoQueryOnVoice, setAutoQueryOnVoice] = useState(Boolean(defaultAutoQueryOnVoice));
  const autoQueryOnVoiceRef = useRef(Boolean(defaultAutoQueryOnVoice));
  const [sessionTimer, setSessionTimer] = useState("0:00");
  const [audioStatus, setAudioStatus] = useState("idle");
  const [ambienceStatus, setAmbienceStatus] = useState("idle");
  const [ambienceTrack, setAmbienceTrack] = useState(null);
  const [ambienceVolume, setAmbienceVolume] = useState(35);
  const [isNarratingScene, setIsNarratingScene] = useState(false);
  const [narrateSceneError, setNarrateSceneError] = useState("");
  const [activeSceneTriggerName, setActiveSceneTriggerName] = useState("");
  const [sceneTriggerError, setSceneTriggerError] = useState("");
  const [sceneSuggestions, setSceneSuggestions] = useState([]);
  const [sceneSuggestionsLoading, setSceneSuggestionsLoading] = useState(false);
  const [sceneSuggestionsError, setSceneSuggestionsError] = useState("");
  const [activeSuggestedSceneId, setActiveSuggestedSceneId] = useState("");
  const [isLaunchingEncounter, setIsLaunchingEncounter] = useState(false);
  const [launchEncounterError, setLaunchEncounterError] = useState("");
  const [assistantListening, setAssistantListening] = useState(false);
  const [assistantAnalyzing, setAssistantAnalyzing] = useState(false);
  const [assistantError, setAssistantError] = useState("");
  const [assistantPartialTranscript, setAssistantPartialTranscript] = useState("");
  const [assistantSuggestions, setAssistantSuggestions] = useState([]);
  const [assistantActionBusyId, setAssistantActionBusyId] = useState("");
  const [ignoredAssistantSuggestionIds, setIgnoredAssistantSuggestionIds] = useState([]);
  const [sceneActionBusy, setSceneActionBusy] = useState("");
  const [sceneActionError, setSceneActionError] = useState("");
  const [npcVoiceModal, setNpcVoiceModal] = useState({
    open: false,
    mode: "speak",
    npc: null,
    value: "",
    busy: false,
    error: "",
    generatedText: "",
  });
  const sessionStartRef = useRef(null);
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const wakeRecognitionRef = useRef(null);
  const wakeRestartTimerRef = useRef(null);
  const wakeCaptureTimeoutRef = useRef(null);
  const assistantRecognitionRef = useRef(null);
  const assistantRestartTimerRef = useRef(null);
  const assistantListeningRef = useRef(false);
  const assistantRecentEntriesRef = useRef([]);
  const assistantPendingEntriesRef = useRef(0);
  const silenceMonitorFrameRef = useRef(null);
  const silenceStartAtRef = useRef(0);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const analyserDataRef = useRef(null);
  const isMicActiveRef = useRef(false);
  const isWakeArmedRef = useRef(false);
  const ambienceAudioRef = useRef(null);
  const ambienceVolumeRef = useRef(0.35);
  const ambienceVolumeTouchedRef = useRef(false);
  const lastActivatedSceneIdRef = useRef("");
  const sceneActivationRequestRef = useRef("");
  const pendingAmbienceTrackRef = useRef(null);
  const stopMicCaptureRef = useRef(() => {});

  const campaign = campaignCtx?.campaign ?? campaignData;
  const scenes = campaign?.scenes?.length ? campaign.scenes : DEFAULT_SCENES;
  const selectedSceneIdx = useSharedCampaign ? campaignCtx.activeSceneIndex : selectedSceneIdxLocal;
  const setSelectedSceneIdx = useSharedCampaign ? campaignCtx.setActiveSceneIndex : setSelectedSceneIdxLocal;
  const fallbackScene = scenes[selectedSceneIdx] || scenes[0] || null;
  const scene = useSharedCampaign ? (campaignCtx.activeScene || fallbackScene) : (scenes[selectedSceneIdxLocal] || scenes[0] || null);
  const activeSessionRecord = useCampaignContextStore((state) => (
    state.activeSessionId
      ? state.sessions.find((session) => session.id === state.activeSessionId) ?? null
      : null
  ));
  const campaignActiveSessionId = String(
    campaign?.activeSessionId
    || campaign?.active_session_id
    || (Array.isArray(campaign?.sessions)
      ? campaign.sessions.find((session) => String(session?.status || "").toLowerCase() === "active")?.id || ""
      : "")
  ).trim();
  const hasActiveSession = useSharedCampaign
    ? Boolean(activeSessionRecord?.id || campaignActiveSessionId)
    : Boolean(campaignActiveSessionId);
  const actionLog = useSharedCampaign ? campaignCtx.actionLog : actionLogLocal;
  const lastPlayerAction = useMemo(() => {
    if (!Array.isArray(actionLog)) return "";
    const recentPlayerEntry = [...actionLog]
      .reverse()
      .find((entry) => String(entry?.role || "").toLowerCase() === "player" && String(entry?.text || "").trim());
    return recentPlayerEntry ? String(recentPlayerEntry.text).trim() : "";
  }, [actionLog]);
  const assistantContext = useMemo(() => buildSessionAssistantContext({
    campaign,
    scene,
    actionLog,
  }), [actionLog, campaign, scene]);
  const visibleAssistantSuggestions = useMemo(() => {
    const suggestions = buildSessionAssistantSuggestions({
      aiSuggestions: assistantSuggestions,
      context: assistantContext,
      scene,
    });
    if (!ignoredAssistantSuggestionIds.length) return suggestions;
    const ignored = new Set(ignoredAssistantSuggestionIds);
    return suggestions.filter((suggestion) => !ignored.has(suggestion.id));
  }, [assistantContext, assistantSuggestions, ignoredAssistantSuggestionIds, scene]);
  const appendActionLogLocal = useCallback((role, text, meta = "") => {
    if (!text) return;
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      role,
      text,
      meta,
    };
    setActionLogLocal((prev) => [...prev, entry].slice(-100));
  }, []);
  const appendActionLog = useSharedCampaign ? campaignCtx.appendActionLog : appendActionLogLocal;

  const appendSessionEntry = useCallback((legacyRole, eventType, text, meta = "") => {
    if (!text) return;
    appendActionLog(legacyRole, text, meta);
    const store = useCampaignContextStore.getState();
    addSessionLogEntry({
      type: eventType,
      text,
      sceneId: store.activeSceneId ?? undefined,
      sessionId: store.activeSessionId ?? undefined,
    });
    persistSessionEvent(authFetch, {
      type: eventType,
      text,
      scene_id: store.activeSceneId,
      session_id: store.activeSessionId,
    });
  }, [appendActionLog, authFetch]);

  const playAudioUrl = useCallback((audioUrl, { revokeOnEnd = true } = {}) => new Promise((resolve, reject) => {
    if (!audioUrl) {
      reject(new Error("No audio returned."));
      return;
    }
    const audio = new Audio(audioUrl);
    setAudioStatus("playing");
    audio.onended = () => {
      setAudioStatus("idle");
      if (revokeOnEnd) URL.revokeObjectURL(audioUrl);
      resolve();
    };
    audio.onerror = () => {
      setAudioStatus("idle");
      if (revokeOnEnd) URL.revokeObjectURL(audioUrl);
      reject(new Error("Audio failed to play."));
    };
    audio.play().catch((error) => {
      setAudioStatus("idle");
      if (revokeOnEnd) URL.revokeObjectURL(audioUrl);
      reject(error);
    });
  }), []);

  const playAudioBlob = useCallback(async (blob, { persistUrl = false } = {}) => {
    const audioUrl = URL.createObjectURL(blob);
    await playAudioUrl(audioUrl, { revokeOnEnd: !persistUrl });
    return audioUrl;
  }, [playAudioUrl]);

  const playBase64Audio = useCallback(async (encoded, mimeType = "audio/wav") => {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    await playAudioBlob(new Blob([bytes], { type: mimeType }));
  }, [playAudioBlob]);

  const stopAmbienceLoop = useCallback(() => {
    const audio = ambienceAudioRef.current;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    pendingAmbienceTrackRef.current = null;
    setAmbienceTrack(null);
    setAmbienceStatus("idle");
  }, []);

  const handleAmbienceVolumeChange = useCallback((value) => {
    const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 35));
    ambienceVolumeTouchedRef.current = true;
    ambienceVolumeRef.current = clamped / 100;
    setAmbienceVolume(clamped);
    const audio = ambienceAudioRef.current;
    if (audio) {
      audio.volume = ambienceVolumeRef.current;
    }
  }, []);

  const playAmbienceLoop = useCallback(async (track) => {
    if (!track?.url) {
      stopAmbienceLoop();
      return;
    }

    let audio = ambienceAudioRef.current;
    if (!audio) {
      audio = new Audio();
      audio.preload = "auto";
      ambienceAudioRef.current = audio;
    }

    const resolvedUrl = typeof window !== "undefined"
      ? new URL(track.url, window.location.origin).toString()
      : track.url;

    if (audio.src !== resolvedUrl) {
      audio.src = resolvedUrl;
      audio.currentTime = 0;
    }

    audio.loop = track.loop !== false;
    const nextVolume = ambienceVolumeTouchedRef.current
      ? ambienceVolumeRef.current
      : typeof track.volume === "number"
        ? track.volume
        : ambienceVolumeRef.current;
    if (!ambienceVolumeTouchedRef.current) {
      ambienceVolumeRef.current = nextVolume;
      setAmbienceVolume(Math.round(nextVolume * 100));
    }
    audio.volume = nextVolume;
    audio.onerror = () => setAmbienceStatus("idle");

    setAmbienceTrack(track);
    setAmbienceStatus("loading");
    try {
      await audio.play();
    } catch (error) {
      pendingAmbienceTrackRef.current = track;
      setAmbienceStatus("idle");
      if (error?.name !== "NotAllowedError") {
        throw error;
      }
      return;
    }
    pendingAmbienceTrackRef.current = null;
    setAmbienceStatus("playing");
  }, [stopAmbienceLoop]);

  useEffect(() => {
    const audio = ambienceAudioRef.current;
    if (audio) {
      audio.volume = ambienceVolumeRef.current;
    }
  }, [ambienceVolume]);

  const syncActivatedSceneState = useCallback((activatedScene, sceneIndex = null) => {
    const resolvedSceneId = String(activatedScene?.id || "").trim();
    if (useSharedCampaign) {
      const store = useCampaignContextStore.getState();
      if (resolvedSceneId) {
        const existingScene = store.scenes.find((item) => item.id === resolvedSceneId) || null;
        if (existingScene) {
          store.upsertScene({
            ...existingScene,
            title: activatedScene?.title || existingScene.title,
            name: activatedScene?.name || existingScene.name,
            summary: activatedScene?.summary || activatedScene?.description || activatedScene?.read_aloud || existingScene.summary,
            description: activatedScene?.description || activatedScene?.read_aloud || activatedScene?.notes || existingScene.description,
            type: activatedScene?.type || existingScene.type,
            location: activatedScene?.location || existingScene.location,
            notes: activatedScene?.notes || existingScene.notes,
            readAloud: activatedScene?.read_aloud || existingScene.readAloud,
            read_aloud: activatedScene?.read_aloud || existingScene.read_aloud,
            atmosphereType: activatedScene?.atmosphere_type || existingScene.atmosphereType,
            atmosphere_type: activatedScene?.atmosphere_type || existingScene.atmosphere_type,
            ambienceTrack: activatedScene?.ambience_track ?? existingScene.ambienceTrack ?? null,
            ambience_track: activatedScene?.ambience_track ?? existingScene.ambience_track ?? null,
          });
        }
        store.setActiveScene(resolvedSceneId);
        const activeStoreSessionId = store.activeSessionId || campaignActiveSessionId;
        if (activeStoreSessionId) {
          const session = store.sessions.find((item) => item.id === activeStoreSessionId);
          if (session) {
            store.upsertSession({ ...session, activeSceneId: resolvedSceneId });
          }
        }
      }
      return;
    }

    if (sceneIndex != null) {
      setSelectedSceneIdxLocal(sceneIndex);
    }
  }, [campaignActiveSessionId, useSharedCampaign]);

  const activateSceneViaBackend = useCallback(async (
    sceneTarget,
    { sceneIndex = null, combat = false, force = false, resetAtmosphereOverride = false } = {},
  ) => {
    if (!sceneTarget) return null;

    const sceneRef = /^\d+$/.test(String(sceneTarget?.id || ""))
      ? String(sceneTarget.id)
      : String(sceneTarget?.title || sceneTarget?.id || "").trim();
    if (!sceneRef) {
      throw new Error("Scene id is missing.");
    }

    if (!hasActiveSession) {
      if (!combat) {
        syncActivatedSceneState(sceneTarget, sceneIndex);
      }
      return { scene: sceneTarget, ambience_audio: null };
    }

    if (!force && !combat && lastActivatedSceneIdRef.current === sceneRef) {
      return { scene: sceneTarget, ambience_audio: ambienceTrack };
    }

    const requestKey = `${combat ? "combat" : "activate"}:${sceneRef}`;
    if (sceneActivationRequestRef.current === requestKey) {
      return null;
    }
    sceneActivationRequestRef.current = requestKey;

    try {
      const response = await authFetch(combat ? "/scene/combat-start" : "/scene/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: sceneRef,
          reset_atmosphere_override: Boolean(resetAtmosphereOverride),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "Scene activation failed.");
      }

      const activatedScene = payload?.scene || sceneTarget;
      const resolvedSceneId = String(activatedScene?.id || sceneRef).trim();
      if (resolvedSceneId) {
        lastActivatedSceneIdRef.current = resolvedSceneId;
      }
      syncActivatedSceneState(activatedScene, sceneIndex);

      if (payload?.ambience_audio?.url) {
        await playAmbienceLoop(payload.ambience_audio);
      } else if (!combat) {
        stopAmbienceLoop();
      }

      return payload;
    } finally {
      if (sceneActivationRequestRef.current === requestKey) {
        sceneActivationRequestRef.current = "";
      }
    }
  }, [
    ambienceTrack,
    authFetch,
    hasActiveSession,
    playAmbienceLoop,
    stopAmbienceLoop,
    syncActivatedSceneState,
  ]);

  useEffect(() => {
    const currentSceneId = String(scene?.id || scene?.title || "").trim();
    if (!currentSceneId) {
      setSceneSuggestions([]);
      setSceneSuggestionsError("");
      setSceneSuggestionsLoading(false);
      return undefined;
    }

    let cancelled = false;
    const loadSuggestions = async () => {
      setSceneSuggestionsLoading(true);
      setSceneSuggestionsError("");
      try {
        const response = await authFetch("/scene/suggestions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            current_scene_id: currentSceneId,
            player_action: lastPlayerAction,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload?.detail || payload?.error || "Could not load scene suggestions.");
        }
        if (!cancelled) {
          setSceneSuggestions(Array.isArray(payload?.scenes) ? payload.scenes : []);
        }
      } catch (error) {
        if (!cancelled) {
          setSceneSuggestions([]);
          setSceneSuggestionsError(error?.message || "Could not load scene suggestions.");
        }
      } finally {
        if (!cancelled) {
          setSceneSuggestionsLoading(false);
        }
      }
    };

    void loadSuggestions();
    return () => {
      cancelled = true;
    };
  }, [authFetch, lastPlayerAction, scene?.id, scene?.title]);

  const handleActivateSuggestedScene = useCallback(async (suggestedScene) => {
    if (!suggestedScene) return;
    const sceneRef = String(suggestedScene?.id || suggestedScene?.title || suggestedScene?.name || "").trim();
    if (!sceneRef) return;

    setActiveSuggestedSceneId(sceneRef);
    setSceneSuggestionsError("");
    try {
      const nextIndex = scenes.findIndex((candidate) => (
        String(candidate?.id || "").trim() === sceneRef
        || String(candidate?.title || "").trim() === String(suggestedScene?.title || "").trim()
      ));
      await activateSceneViaBackend(suggestedScene, {
        sceneIndex: nextIndex >= 0 ? nextIndex : null,
        force: true,
      });
      appendActionLog(
        "assistant",
        `Suggested next scene activated: ${suggestedScene?.title || suggestedScene?.name || "Scene"}.`,
        "Scene Suggestions",
      );
    } catch (error) {
      setSceneSuggestionsError(error?.message || "Could not activate suggested scene.");
    } finally {
      setActiveSuggestedSceneId("");
    }
  }, [activateSceneViaBackend, appendActionLog, scenes]);

  const resolveNarrationVoiceId = useCallback(async () => {
    const response = await authFetch("/voices/list");
    if (!response.ok) {
      throw new Error("Could not load available voices.");
    }
    const voices = await response.json();
    const firstVoice = Array.isArray(voices) ? voices.find((voice) => voice?.voice_id) : null;
    if (!firstVoice?.voice_id) {
      throw new Error("No voice available for narration.");
    }
    return firstVoice.voice_id;
  }, [authFetch]);

  const narrateText = useCallback(async (text, meta = "Session Assistant") => {
    const narrationText = String(text || "").trim();
    if (!narrationText) return;

    setAudioStatus("loading");
    const voiceId = scene?.narrator_voice_id || scene?.narratorVoiceId || await resolveNarrationVoiceId();
    const response = await authFetch("/tts/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: narrationText, voice_id: voiceId }),
    });
    if (!response.ok) {
      throw new Error((await response.text()) || "Narration failed.");
    }
    const blob = await response.blob();
    appendSessionEntry("assistant", "narration", narrationText, meta);
    await playAudioBlob(blob);
  }, [appendSessionEntry, authFetch, playAudioBlob, resolveNarrationVoiceId, scene?.narratorVoiceId, scene?.narrator_voice_id]);

  const runSceneBrainAction = useCallback(async (mode) => {
    if (!scene) return;

    const sceneTitle = scene?.title || "Current scene";
    const npcNames = assistantContext.npcsInScene.map((npc) => npc.name).join(", ");
    const questNames = assistantContext.activeQuests.map((quest) => quest.name).join(", ");
    const recentEvents = assistantContext.recentEvents.slice(0, 3).map((entry) => entry.text).join(" | ");

    const prompt = mode === "twist"
      ? [
          `Give the GM one strong twist or complication for the scene "${sceneTitle}".`,
          assistantContext.currentLocation ? `Location: ${assistantContext.currentLocation}.` : "",
          npcNames ? `NPCs present: ${npcNames}.` : "",
          questNames ? `Active quests: ${questNames}.` : "",
          recentEvents ? `Recent events: ${recentEvents}.` : "",
          "Return only 2-3 concise sentences the GM can use immediately.",
        ].filter(Boolean).join(" ")
      : [
          `Expand the scene "${sceneTitle}" into a vivid read-aloud description for the GM.`,
          assistantContext.currentLocation ? `Location: ${assistantContext.currentLocation}.` : "",
          npcNames ? `NPCs present: ${npcNames}.` : "",
          questNames ? `Active quests: ${questNames}.` : "",
          recentEvents ? `Recent events: ${recentEvents}.` : "",
          "Keep it to 3-4 sentences. Return only the description text.",
        ].filter(Boolean).join(" ");

    setSceneActionBusy(mode);
    setSceneActionError("");
    try {
      const response = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: prompt }),
      });
      if (!response.ok) {
        throw new Error((await response.text()) || "Scene assistant action failed.");
      }
      const payload = await response.json().catch(() => ({}));
      const content = String(payload?.content || payload?.text || "").trim();
      if (!content) {
        throw new Error("Scene assistant returned no usable text.");
      }
      appendSessionEntry(
        "assistant",
        mode === "twist" ? "system" : "narration",
        content,
        mode === "twist" ? "Add Twist" : "Expand Description",
      );
    } catch (error) {
      setSceneActionError(error?.message || "Scene assistant action failed.");
    } finally {
      setSceneActionBusy("");
    }
  }, [appendSessionEntry, assistantContext, authFetch, scene]);

  const getAssistantContextPayload = useCallback((transcriptEntries) => {
    const recentEntries = (transcriptEntries || []).filter(Boolean).slice(-8);
    const sceneNpcs = assistantContext.npcsInScene.map((npc) => npc.raw).filter(Boolean);
    return {
      transcript_entries: recentEntries,
      scene_title: scene?.title || "",
      scene_summary: scene?.read_aloud || scene?.summary || scene?.notes || "",
      location_name: assistantContext.currentLocation || "",
      active_quests: assistantContext.activeQuests.map((quest) => quest.name),
      recent_events: assistantContext.recentEvents.map((entry) => entry.text),
      npcs: sceneNpcs.map((npc) => ({
        id: npc.id,
        name: npc.name,
        role: npc.role,
        description: npc.description || npc.personality || npc.summary || "",
      })),
    };
  }, [assistantContext, scene]);

  const runSessionAssistantAnalysis = useCallback(async (transcriptEntries, { force = false } = {}) => {
    const entries = (transcriptEntries || []).filter(Boolean).slice(-8);
    if (!entries.length) return;
    if (!force && entries.length < 3) return;

    setAssistantAnalyzing(true);
    setAssistantError("");

    try {
      const response = await authFetch("/session-assistant/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getAssistantContextPayload(entries)),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "Session assistant analysis failed.");
      }
      const suggestions = Array.isArray(payload?.suggestions)
        ? payload.suggestions.map((suggestion, index) => ({
            ...suggestion,
            id: suggestion.id || `${suggestion.type || "suggestion"}-${index}`,
          }))
        : [];
      setAssistantSuggestions(suggestions);
    } catch (error) {
      setAssistantError(error?.message || "Session assistant analysis failed.");
    } finally {
      setAssistantAnalyzing(false);
    }
  }, [authFetch, getAssistantContextPayload]);

  const stopAssistantListening = useCallback(() => {
    if (assistantRestartTimerRef.current) {
      clearTimeout(assistantRestartTimerRef.current);
      assistantRestartTimerRef.current = null;
    }
    assistantListeningRef.current = false;
    setAssistantListening(false);
    setAssistantPartialTranscript("");
    const recognition = assistantRecognitionRef.current;
    if (recognition) {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      assistantRecognitionRef.current = null;
      try {
        recognition.stop();
      } catch {
        /* no-op */
      }
    }
  }, []);

  useEffect(() => () => {
    stopAssistantListening();
  }, [stopAssistantListening]);

  const startAssistantListening = useCallback(() => {
    if (typeof window === "undefined") return;
    if (assistantListeningRef.current) return;
    if (isMicActiveRef.current) {
      setAssistantError("Stop live microphone capture before starting Session Assistant.");
      return;
    }

    const SpeechRecognitionApi = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionApi) {
      setAssistantError("Browser speech recognition is not supported in this browser.");
      return;
    }

    setAssistantError("");
    setAssistantPartialTranscript("");
    assistantListeningRef.current = true;
    setAssistantListening(true);

    const bootRecognition = () => {
      if (!assistantListeningRef.current) return;

      const recognition = new SpeechRecognitionApi();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event) => {
        let partial = "";

        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const transcript = String(event.results[i]?.[0]?.transcript || "").replace(/\s+/g, " ").trim();
          if (!transcript) continue;

          if (event.results[i].isFinal) {
            appendSessionEntry("player", "player", transcript, "Session Assistant");
            const nextEntries = [...assistantRecentEntriesRef.current, transcript].slice(-8);
            assistantRecentEntriesRef.current = nextEntries;
            assistantPendingEntriesRef.current += 1;
            if (assistantPendingEntriesRef.current >= 3) {
              assistantPendingEntriesRef.current = 0;
              void runSessionAssistantAnalysis(nextEntries, { force: true });
            }
          } else {
            partial = partial ? `${partial} ${transcript}` : transcript;
          }
        }

        setAssistantPartialTranscript(partial);
      };

      recognition.onerror = (event) => {
        if (event.error === "no-speech" || event.error === "aborted") return;
        setAssistantError(`Listening error: ${event.error}`);
      };

      recognition.onend = () => {
        assistantRecognitionRef.current = null;
        setAssistantPartialTranscript("");
        if (!assistantListeningRef.current) {
          setAssistantListening(false);
          return;
        }
        assistantRestartTimerRef.current = setTimeout(() => {
          bootRecognition();
        }, 350);
      };

      assistantRecognitionRef.current = recognition;
      try {
        recognition.start();
      } catch (error) {
        assistantRecognitionRef.current = null;
        assistantListeningRef.current = false;
        setAssistantListening(false);
        setAssistantError(error?.message || "Failed to start session assistant listening.");
      }
    };

    bootRecognition();
  }, [appendSessionEntry, runSessionAssistantAnalysis]);

  const handleNarrateScene = useCallback(async (activeScene) => {
    const sceneTarget = activeScene || scene;
    const sceneText = (sceneTarget?.read_aloud || sceneTarget?.notes || "").trim();
    if (!sceneTarget || !sceneText) return;

    setIsNarratingScene(true);
    setNarrateSceneError("");
    setAudioStatus("loading");

    try {
      let response;
      const backendCampaignId = getBackendCampaignId();
      const sceneRef = /^\d+$/.test(String(sceneTarget?.id || ""))
        ? String(sceneTarget.id)
        : String(sceneTarget?.title || sceneTarget?.id || "").trim();
      if (backendCampaignId && sceneRef) {
        response = await authFetch("/tts/narrate-scene", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scene_id: sceneRef }),
        });
      } else {
        const voiceId = await resolveNarrationVoiceId();
        response = await authFetch("/tts/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: sceneText, voice_id: voiceId }),
        });
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Narration failed.");
      }

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      appendSessionEntry("assistant", "narration", sceneText, "Scene Narration");

      useCampaignContextStore.getState().addNarrationClip({
        title: sceneTarget?.title || "Scene Narration",
        voiceId: sceneTarget?.narrator_voice_id,
        audioUrl,
      });
      await playAudioUrl(audioUrl, { revokeOnEnd: false });
    } catch (error) {
      setNarrateSceneError(error?.message || "Narration failed.");
      setAudioStatus("idle");
    } finally {
      setIsNarratingScene(false);
    }
  }, [appendSessionEntry, authFetch, playAudioUrl, resolveNarrationVoiceId, scene]);

  const handleSceneTrigger = useCallback(async (trigger) => {
    const sceneTarget = scene;
    if (!sceneTarget || !trigger?.name || activeSceneTriggerName) return;

    setActiveSceneTriggerName(trigger.name);
    setSceneTriggerError("");
    setAudioStatus("loading");

    try {
      const backendCampaignId = getBackendCampaignId();
      if (!backendCampaignId) {
        if (String(trigger.type || "").toLowerCase() === "narration") {
          const narrationText = String(trigger.text || sceneTarget?.read_aloud || sceneTarget?.notes || "").trim();
          if (!narrationText) {
            await handleNarrateScene(sceneTarget);
            return;
          }

          const voiceId = sceneTarget?.narrator_voice_id || await resolveNarrationVoiceId();
          const response = await authFetch("/tts/narrate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: narrationText, voice_id: voiceId }),
          });
          if (!response.ok) {
            throw new Error((await response.text()) || "Narration failed.");
          }

          const blob = await response.blob();
          appendSessionEntry("assistant", "narration", narrationText, trigger.name);
          await playAudioBlob(blob);
          return;
        }
        throw new Error("Scene control requires a campaign loaded from the backend.");
      }

      const sceneRef = /^\d+$/.test(String(sceneTarget?.id || ""))
        ? String(sceneTarget.id)
        : String(sceneTarget?.title || sceneTarget?.id || "").trim();
      if (!sceneRef) {
        throw new Error("Scene id is missing for this trigger.");
      }

      const response = await authFetch("/scene/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: sceneRef,
          trigger_name: String(trigger.name),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "Scene trigger failed.");
      }

      const triggerText = String(payload?.text || "").trim();
      const eventType = payload?.event_type || (
        payload?.log_type === "narration"
          ? "narration"
          : payload?.log_type === "npc"
            ? "npc"
            : "system"
      );
      const displayText = String(payload?.display_text || "").trim()
        || (eventType === "npc" && payload?.npc_name ? `${payload.npc_name}: ${triggerText}` : triggerText);

      if (displayText) {
        appendSessionEntry(
          "assistant",
          eventType,
          displayText,
          payload?.trigger_name || trigger.name
        );
      }

      if (payload.audio_base64) {
        await playBase64Audio(payload.audio_base64, payload.mime_type || "audio/wav");
      }
      if (!payload.audio_base64) {
        setAudioStatus("idle");
      }
    } catch (error) {
      setSceneTriggerError(error?.message || "Scene trigger failed.");
      setAudioStatus("idle");
    } finally {
      setActiveSceneTriggerName("");
    }
  }, [
    activeSceneTriggerName,
    appendSessionEntry,
    authFetch,
    handleNarrateScene,
    playAudioBlob,
    playBase64Audio,
    resolveNarrationVoiceId,
    scene,
  ]);

  const handleLaunchEncounter = useCallback(async () => {
    const sceneTarget = scene;
    if (!sceneTarget || isLaunchingEncounter) return;

    const encounterRef = resolveEncounterRef(sceneTarget);
    if (!encounterRef) {
      setLaunchEncounterError("Encounter id is missing for the active scene.");
      return;
    }

    setIsLaunchingEncounter(true);
    setLaunchEncounterError("");
    setAudioStatus("loading");

    try {
      const response = await authFetch("/encounter/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ encounter_id: encounterRef }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "Encounter launch failed.");
      }

      const activatedScene = payload?.scene || sceneTarget;
      const activatedSceneId = String(activatedScene?.id || "").trim();
      if (activatedSceneId) {
        lastActivatedSceneIdRef.current = activatedSceneId;
        const nextIndex = scenes.findIndex((candidate) => (
          String(candidate?.id || "").trim() === activatedSceneId
          || String(candidate?.title || "").trim() === String(activatedScene?.title || "").trim()
        ));
        syncActivatedSceneState(activatedScene, nextIndex >= 0 ? nextIndex : null);
      }

      if (payload?.ambience_audio?.url) {
        await playAmbienceLoop(payload.ambience_audio);
      }

      const narrationText = String(
        payload?.narration_audio?.text
        || payload?.encounter?.intro_text
        || ""
      ).trim();
      if (narrationText) {
        appendSessionEntry("assistant", "narration", narrationText, payload?.encounter?.name || "Encounter");
      }
      if (payload?.narration_audio?.audio_base64) {
        await playBase64Audio(
          payload.narration_audio.audio_base64,
          payload.narration_audio.mime_type || "audio/wav",
        );
      }

      const enemyText = String(
        payload?.enemy_dialogue_audio?.text
        || payload?.enemy_dialogue_text
        || ""
      ).trim();
      const enemyName = String(
        payload?.enemy_dialogue_audio?.npc_name
        || payload?.enemy_npc_name
        || "Enemy"
      ).trim();
      if (enemyText) {
        appendSessionEntry(
          "assistant",
          "npc",
          enemyName ? `${enemyName}: ${enemyText}` : enemyText,
          payload?.encounter?.name || "Encounter",
        );
      }
      if (payload?.enemy_dialogue_audio?.audio_base64) {
        await playBase64Audio(
          payload.enemy_dialogue_audio.audio_base64,
          payload.enemy_dialogue_audio.mime_type || "audio/wav",
        );
      }

      if (!payload?.narration_audio?.audio_base64 && !payload?.enemy_dialogue_audio?.audio_base64) {
        setAudioStatus("idle");
      }
    } catch (error) {
      setLaunchEncounterError(error?.message || "Encounter launch failed.");
      setAudioStatus("idle");
    } finally {
      setIsLaunchingEncounter(false);
    }
  }, [
    appendSessionEntry,
    authFetch,
    isLaunchingEncounter,
    playAmbienceLoop,
    playBase64Audio,
    scene,
    scenes,
    syncActivatedSceneState,
  ]);

  const handleSelectScene = useCallback((nextIndex) => {
    const targetScene = scenes[nextIndex] || null;
    if (!targetScene) return;

    if (!hasActiveSession) {
      setSelectedSceneIdx(nextIndex);
      return;
    }

    void activateSceneViaBackend(targetScene, { sceneIndex: nextIndex }).catch((error) => {
      appendActionLog("error", error?.message || "Scene activation failed.", "Atmosphere");
    });
  }, [activateSceneViaBackend, appendActionLog, hasActiveSession, scenes, setSelectedSceneIdx]);

  const handleCombatStart = useCallback((targetScene) => {
    if (!targetScene) return;
    void activateSceneViaBackend(targetScene, {
      sceneIndex: selectedSceneIdx,
      combat: true,
      force: true,
    }).then((payload) => {
      if (payload?.ambience_audio?.label) {
        appendActionLog("assistant", `${payload.ambience_audio.label} engaged.`, "Atmosphere");
      }
    }).catch((error) => {
      appendActionLog("error", error?.message || "Combat ambience failed.", "Atmosphere");
    });
  }, [activateSceneViaBackend, appendActionLog, selectedSceneIdx]);

  const handleResumeSceneAmbience = useCallback((targetScene) => {
    if (!targetScene) return;
    void activateSceneViaBackend(targetScene, {
      sceneIndex: selectedSceneIdx,
      force: true,
      resetAtmosphereOverride: true,
    }).then((payload) => {
      if (payload?.ambience_audio?.label) {
        appendActionLog("assistant", `${payload.ambience_audio.label} resumed.`, "Atmosphere");
      }
    }).catch((error) => {
      appendActionLog("error", error?.message || "Scene ambience failed.", "Atmosphere");
    });
  }, [activateSceneViaBackend, appendActionLog, selectedSceneIdx]);

  useEffect(() => {
    setSceneTriggerError("");
    setActiveSceneTriggerName("");
    setSceneActionBusy("");
    setSceneActionError("");
  }, [scene?.id, scene?.title]);

  useEffect(() => {
    if (!hasActiveSession || !scene) {
      lastActivatedSceneIdRef.current = "";
      stopAmbienceLoop();
      return;
    }

    const resolvedSceneId = String(scene?.id || scene?.title || "").trim();
    if (!resolvedSceneId || lastActivatedSceneIdRef.current === resolvedSceneId) {
      return;
    }

    void activateSceneViaBackend(scene, {
      sceneIndex: selectedSceneIdx,
      force: true,
    }).catch((error) => {
      appendActionLog("error", error?.message || "Scene activation failed.", "Atmosphere");
    });
  }, [
    activateSceneViaBackend,
    appendActionLog,
    hasActiveSession,
    scene,
    selectedSceneIdx,
    stopAmbienceLoop,
  ]);

  useEffect(() => () => {
    const audio = ambienceAudioRef.current;
    if (audio) {
      audio.pause();
      ambienceAudioRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const resumeAmbience = () => {
      const pendingTrack = pendingAmbienceTrackRef.current;
      if (!pendingTrack) return;
      void playAmbienceLoop(pendingTrack);
    };

    window.addEventListener("pointerdown", resumeAmbience);
    window.addEventListener("keydown", resumeAmbience);
    return () => {
      window.removeEventListener("pointerdown", resumeAmbience);
      window.removeEventListener("keydown", resumeAmbience);
    };
  }, [playAmbienceLoop]);

  const handleNarrateCampaignAnswer = useCallback(async ({ campaignId, answer }) => {
    const text = (answer || "").trim();
    if (!campaignId || !text) {
      throw new Error("Campaign answer narration requires a saved campaign and answer text.");
    }

    setAudioStatus("loading");
    const response = await authFetch("/tts/narrate-answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        campaign_id: campaignId,
        answer: text,
      }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      setAudioStatus("idle");
      throw new Error(payload?.detail || payload?.error || "Campaign answer narration failed.");
    }

    const blob = await response.blob();
    await playAudioBlob(blob);
  }, [authFetch, playAudioBlob]);

  const openNpcVoiceModal = useCallback((npc, mode) => {
    if (!npc) return;
    setSelectedNpcName(npc.name || null);
    setNpcVoiceModal({
      open: true,
      mode,
      npc,
      value: "",
      busy: false,
      error: "",
      generatedText: "",
    });
  }, []);

  const closeNpcVoiceModal = useCallback(() => {
    setNpcVoiceModal((current) => (
      current.busy
        ? current
        : { open: false, mode: "speak", npc: null, value: "", busy: false, error: "", generatedText: "" }
    ));
  }, []);

  const submitNpcVoiceModal = useCallback(async () => {
    const npc = npcVoiceModal.npc;
    const value = (npcVoiceModal.value || "").trim();
    if (!npc || !value || npcVoiceModal.busy) return;

    const npcId = npc.id || npc.name;
    setNpcVoiceModal((current) => ({ ...current, busy: true, error: "" }));
    setAudioStatus("loading");

    try {
      if (npcVoiceModal.mode === "generate") {
        appendSessionEntry("player", "player", value, `${npc.name || "NPC"} Prompt`);
        const response = await authFetch("/npc/generate-dialogue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ npc_id: String(npcId), player_input: value, scene_id: scene?.id }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload?.detail || payload?.error || "NPC response failed.");
        }

        const generatedLine = `${npc.name || "NPC"}: ${payload.generated_text || ""}`.trim();
        appendSessionEntry("assistant", "npc", generatedLine, "AI NPC Response");
        if (payload.audio_base64) {
          await playBase64Audio(payload.audio_base64, payload.mime_type || "audio/wav");
        }
        setNpcVoiceModal((current) => ({
          ...current,
          busy: false,
          error: "",
          generatedText: payload.generated_text || "",
        }));
        return;
      }

      const response = await authFetch("/tts/npc-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: String(npcId), text: value }),
      });
      if (!response.ok) {
        throw new Error((await response.text()) || "NPC dialogue failed.");
      }
      const blob = await response.blob();
      appendSessionEntry("assistant", "npc", `${npc.name || "NPC"}: ${value}`, "NPC Voice");
      await playAudioBlob(blob);
      setNpcVoiceModal({ open: false, mode: "speak", npc: null, value: "", busy: false, error: "", generatedText: "" });
    } catch (error) {
      setAudioStatus("idle");
      setNpcVoiceModal((current) => ({
        ...current,
        busy: false,
        error: error?.message || "NPC voice action failed.",
      }));
    }
  }, [appendSessionEntry, authFetch, npcVoiceModal, playAudioBlob, playBase64Audio, scene?.id]);

  const preferredPaletteNpc = resolvePreferredNpc(campaign, scene, selectedNpcName);

  useEffect(() => {
    if (!onRegisterCommandActions) return;
    if (!hasActiveSession) {
      onRegisterCommandActions(null);
      return;
    }

    const sceneNarrationText = String(scene?.read_aloud || scene?.notes || "").trim();
    onRegisterCommandActions({
      preferredNpcName: preferredPaletteNpc?.name || "",
      sceneTitle: scene?.title || "",
      hasNarrationText: Boolean(sceneNarrationText),
      narrateScene: () => handleNarrateScene(scene),
      speakAsNpc: () => {
        if (!preferredPaletteNpc) return;
        openNpcVoiceModal(preferredPaletteNpc, "speak");
      },
      generateNpcDialogue: () => {
        if (!preferredPaletteNpc) return;
        openNpcVoiceModal(preferredPaletteNpc, "generate");
      },
    });
  }, [
    handleNarrateScene,
    hasActiveSession,
    onRegisterCommandActions,
    openNpcVoiceModal,
    preferredPaletteNpc,
    scene,
  ]);

  useEffect(() => () => {
    onRegisterCommandActions?.(null);
  }, [onRegisterCommandActions]);

  useEffect(() => {
    const startedAt = activeSessionRecord?.startedAt ? new Date(activeSessionRecord.startedAt).getTime() : NaN;
    sessionStartRef.current = Number.isFinite(startedAt) && startedAt > 0 ? startedAt : Date.now();
    const tick = () => setSessionTimer(formatSessionTimer(sessionStartRef.current));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [activeSessionRecord?.startedAt, hasActiveSession]);

  useEffect(() => {
    if (setBannerState) {
      const bannerAudioStatus = audioStatus !== "idle" ? audioStatus : ambienceStatus;
      setBannerState((prev) => ({
        ...prev,
        sessionTime: sessionTimer,
        activeScene: scene?.title ?? "—",
        audioStatus: bannerAudioStatus,
      }));
    }
  }, [ambienceStatus, sessionTimer, scene?.title, audioStatus, setBannerState]);

  useEffect(() => {
    isMicActiveRef.current = isMicActive;
  }, [isMicActive]);

  useEffect(() => {
    isWakeArmedRef.current = isWakeArmed;
  }, [isWakeArmed]);

  useEffect(() => {
    autoQueryOnVoiceRef.current = autoQueryOnVoice;
  }, [autoQueryOnVoice]);

  useEffect(() => {
    assistantListeningRef.current = assistantListening;
  }, [assistantListening]);

  useEffect(() => {
    setAutoQueryOnVoice(Boolean(defaultAutoQueryOnVoice));
  }, [defaultAutoQueryOnVoice]);

  useEffect(() => {
    assistantRecentEntriesRef.current = [];
    assistantPendingEntriesRef.current = 0;
    setIgnoredAssistantSuggestionIds([]);
    setAssistantSuggestions([]);
    setAssistantPartialTranscript("");
    setAssistantError("");
  }, [scene?.id, scene?.title]);

  useEffect(() => {
    if (!hasActiveSession || !scene) return undefined;

    const transcriptEntries = assistantContext.recentEvents
      .slice(0, 6)
      .map((entry) => String(entry?.text || "").trim())
      .filter(Boolean)
      .reverse();

    if (!transcriptEntries.length && !scene?.summary && !scene?.read_aloud) return undefined;

    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (cancelled) return;
      void runSessionAssistantAnalysis(
        transcriptEntries.length ? transcriptEntries : [scene?.summary || scene?.read_aloud || scene?.title || "Current scene context"],
        { force: true },
      );
    }, 700);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    assistantContext,
    hasActiveSession,
    runSessionAssistantAnalysis,
    scene?.id,
    scene?.read_aloud,
    scene?.summary,
    scene?.title,
  ]);

  useEffect(() => () => {
    stopAssistantListening();
  }, [stopAssistantListening]);

  const stopSilenceMonitoring = useCallback(() => {
    if (silenceMonitorFrameRef.current) {
      cancelAnimationFrame(silenceMonitorFrameRef.current);
      silenceMonitorFrameRef.current = null;
    }
    silenceStartAtRef.current = 0;
    analyserRef.current = null;
    analyserDataRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
  }, []);

  const startSilenceMonitoring = useCallback((stream) => {
    if (typeof window === "undefined") return;
    const AudioContextApi = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextApi) return;

    stopSilenceMonitoring();

    const audioContext = new AudioContextApi();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    const buffer = new Uint8Array(analyser.fftSize);
    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    analyserDataRef.current = buffer;

    const monitor = () => {
      if (!isMicActiveRef.current) return;
      const currentAnalyser = analyserRef.current;
      const data = analyserDataRef.current;
      if (!currentAnalyser || !data) return;

      currentAnalyser.getByteTimeDomainData(data);
      let sumSquares = 0;
      for (let i = 0; i < data.length; i += 1) {
        const centered = (data[i] - 128) / 128;
        sumSquares += centered * centered;
      }
      const rms = Math.sqrt(sumSquares / data.length);
      const now = Date.now();

      if (rms < SILENCE_RMS_THRESHOLD) {
        if (!silenceStartAtRef.current) {
          silenceStartAtRef.current = now;
        } else if (now - silenceStartAtRef.current >= SILENCE_HOLD_MS) {
          appendActionLog("assistant", "Silence detected. Stopping microphone capture.", "STT");
          stopMicCaptureRef.current();
          return;
        }
      } else {
        silenceStartAtRef.current = 0;
      }

      silenceMonitorFrameRef.current = requestAnimationFrame(monitor);
    };

    silenceMonitorFrameRef.current = requestAnimationFrame(monitor);
  }, [appendActionLog, stopSilenceMonitoring]);

  const stopMediaStream = useCallback(() => {
    const stream = mediaStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    stopSilenceMonitoring();
  }, [stopSilenceMonitoring]);

  const stopMicCapture = useCallback(() => {
    if (wakeCaptureTimeoutRef.current) {
      clearTimeout(wakeCaptureTimeoutRef.current);
      wakeCaptureTimeoutRef.current = null;
    }
    setLiveTranscript("");
    isMicActiveRef.current = false;
    stopSilenceMonitoring();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      return;
    }
    stopMediaStream();
    setIsMicActive(false);
  }, [stopMediaStream, stopSilenceMonitoring]);

  useEffect(() => {
    stopMicCaptureRef.current = stopMicCapture;
  }, [stopMicCapture]);

  const startMicCapture = useCallback(async ({ fromWakeWord = false } = {}) => {
    if (isMicActive) return;
    if (assistantListeningRef.current) {
      stopAssistantListening();
      setAssistantError("Session Assistant listening paused while live microphone capture is active.");
    }
    setMicError("");
    setLiveTranscript("");

    if (!navigator?.mediaDevices?.getUserMedia) {
      setMicError("Microphone capture is not supported in this browser.");
      return;
    }

    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setMicError("WebSocket is not connected. Wait for Connected status.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const mimeCandidates = ["audio/webm;codecs=opus", "audio/webm"];
      const preferredMimeType = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      ws.send(JSON.stringify({
        type: "audio_start",
        mime_type: recorder.mimeType || preferredMimeType || "audio/webm",
      }));

      recorder.ondataavailable = async (event) => {
        if (!event.data || event.data.size === 0) return;
        const activeSocket = socketRef.current;
        if (!activeSocket || activeSocket.readyState !== WebSocket.OPEN) return;
        try {
          const chunk = await event.data.arrayBuffer();
          activeSocket.send(chunk);
        } catch (error) {
          setMicError(error?.message || "Failed to stream microphone chunk.");
        }
      };

      recorder.onstop = () => {
        const activeSocket = socketRef.current;
        if (activeSocket && activeSocket.readyState === WebSocket.OPEN) {
          activeSocket.send(JSON.stringify({ type: "audio_end" }));
        }
        stopMediaStream();
        mediaRecorderRef.current = null;
        isMicActiveRef.current = false;
        setIsMicActive(false);
      };

      recorder.onerror = () => {
        setMicError("Microphone recorder error.");
      };

      recorder.start(400);
      isMicActiveRef.current = true;
      setIsMicActive(true);
      startSilenceMonitoring(stream);
      appendActionLog("assistant", "Microphone capture started. Speak now.", "STT");

      if (fromWakeWord) {
        wakeCaptureTimeoutRef.current = setTimeout(() => {
          stopMicCapture();
        }, 12000);
      }
    } catch (error) {
      stopMediaStream();
      mediaRecorderRef.current = null;
      isMicActiveRef.current = false;
      setLiveTranscript("");
      setMicError(error?.message || "Microphone access failed.");
    }
  }, [appendActionLog, isMicActive, startSilenceMonitoring, stopAssistantListening, stopMediaStream, stopMicCapture]);

  const stopWakeRecognition = useCallback(() => {
    if (wakeRestartTimerRef.current) {
      clearTimeout(wakeRestartTimerRef.current);
      wakeRestartTimerRef.current = null;
    }
    const recognition = wakeRecognitionRef.current;
    if (recognition) {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      wakeRecognitionRef.current = null;
      try {
        recognition.stop();
      } catch {
        /* no-op */
      }
    }
  }, []);

  const startWakeRecognition = useCallback(() => {
    if (typeof window === "undefined") return;
    if (!isWakeArmedRef.current || isMicActiveRef.current || wakeRecognitionRef.current) return;

    const SpeechRecognitionApi = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionApi) {
      setWakeError("Wake phrase listener is not supported in this browser.");
      setIsWakeArmed(false);
      return;
    }

    const recognition = new SpeechRecognitionApi();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = normalizeWakeText(event.results[i]?.[0]?.transcript || "");
        if (transcript.includes(WAKE_WORD)) {
          appendActionLog("assistant", `Wake phrase detected: "${WAKE_WORD}".`, "Wake");
          setWakeError("");
          stopWakeRecognition();
          startMicCapture({ fromWakeWord: true });
          break;
        }
      }
    };

    recognition.onerror = (event) => {
      if (event.error === "no-speech" || event.error === "aborted") return;
      setWakeError(`Wake listener error: ${event.error}`);
    };

    recognition.onend = () => {
      wakeRecognitionRef.current = null;
      if (!isWakeArmedRef.current || isMicActiveRef.current) return;
      wakeRestartTimerRef.current = setTimeout(() => {
        startWakeRecognition();
      }, 350);
    };

    wakeRecognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (error) {
      wakeRecognitionRef.current = null;
      setWakeError(error?.message || "Failed to start wake phrase listener.");
    }
  }, [appendActionLog, startMicCapture, stopWakeRecognition]);

  const renderBrainPayload = useCallback((payload) => {
    if (!payload || typeof payload !== "object") {
      appendActionLog("error", "Received malformed response from Co-DM.");
      return;
    }
    if (payload.type === "status") {
      const content = (payload.content || "").trim();
      if (content === "listening-live") {
        appendActionLog("assistant", "Live streaming transcription active.", "STT");
      } else if (content === "listening") {
        appendActionLog("assistant", "Microphone listening started (fallback mode).", "STT");
      } else if (content) {
        appendActionLog("assistant", content, "STT");
      }
      return;
    }
    if (payload.type === "transcript") {
      const transcript = (payload.content || "").trim();
      if (payload.final) {
        setLiveTranscript("");
        if (transcript) {
          appendActionLog("player", transcript, "STT");
          if (autoQueryOnVoiceRef.current) {
            const ws = socketRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: "query", text: transcript }));
            }
          }
        } else {
          appendActionLog("error", "No speech recognized from microphone audio.");
        }
      } else {
        setLiveTranscript(transcript);
      }
      return;
    }
    if (payload.type === "error") {
      appendActionLog("error", payload.content || "Unknown Co-DM error");
      return;
    }
    if (payload.type === "stat_block") {
      const meta = [payload.intent, payload.sources?.length ? `${payload.sources.length} sources` : ""].filter(Boolean).join(" • ");
      appendActionLog("stat_block", payload.content || "", meta);
      addSessionLogEntry({ type: "assistant", text: payload.content || "" });
      persistSessionEvent(authFetch, { type: "assistant", text: payload.content || "" });
      return;
    }
    if (payload.type === "lore") {
      const meta = [payload.intent, payload.sources?.length ? `${payload.sources.length} sources` : ""].filter(Boolean).join(" • ");
      appendActionLog("lore", payload.content || "", meta);
      addSessionLogEntry({ type: "assistant", text: payload.content || "" });
      persistSessionEvent(authFetch, { type: "assistant", text: payload.content || "" });
      return;
    }
    const content = typeof payload.content === "string" ? payload.content : JSON.stringify(payload.content || payload);
    const parts = [];
    if (payload.intent) parts.push(payload.intent);
    if (Array.isArray(payload.sources) && payload.sources.length) parts.push(`${payload.sources.length} sources`);
    appendActionLog("assistant", content, parts.join(" • "));
    addSessionLogEntry({ type: "assistant", text: content });
    persistSessionEvent(authFetch, { type: "assistant", text: content });
  }, [appendActionLog, authFetch]);

  const analyzeAssistantNow = useCallback(() => {
    const fallbackEntries = assistantContext.recentEvents
      .slice(0, 6)
      .map((entry) => String(entry?.text || "").trim())
      .filter(Boolean)
      .reverse();
    void runSessionAssistantAnalysis(
      assistantRecentEntriesRef.current.length ? assistantRecentEntriesRef.current : fallbackEntries,
      { force: true },
    );
  }, [assistantContext.recentEvents, runSessionAssistantAnalysis]);

  const handleNpcWhisper = useCallback(async (npc, { busyId = "" } = {}) => {
    if (!npc) return;
    const prompt = [
      "Offer a brief in-character whisper or private aside for the GM.",
      scene?.title ? `Scene: ${scene.title}.` : "",
      assistantContext.currentLocation ? `Location: ${assistantContext.currentLocation}.` : "",
      "Keep it to one or two sentences.",
    ].filter(Boolean).join(" ");

    setAssistantActionBusyId(String(busyId || npc?.id || npc?.name || "npc-whisper"));
    setAssistantError("");
    try {
      const response = await authFetch("/npc/generate-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          npc_id: String(npc.id || npc.name),
          player_input: prompt,
          scene_id: scene?.id,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload?.detail || payload?.error || "NPC whisper failed.");
      }
      const generatedLine = `${npc.name || "NPC"}: ${payload.generated_text || ""}`.trim();
      appendSessionEntry("assistant", "npc", generatedLine, "Whisper");
      if (payload.audio_base64) {
        await playBase64Audio(payload.audio_base64, payload.mime_type || "audio/wav");
      }
    } catch (error) {
      setAssistantError(error?.message || "NPC whisper failed.");
    } finally {
      setAssistantActionBusyId("");
    }
  }, [appendSessionEntry, assistantContext.currentLocation, authFetch, playBase64Audio, scene?.id, scene?.title]);

  const handleAssistantSuggestionAction = useCallback(async (suggestion, mode = "run") => {
    if (!suggestion || !suggestion.id || assistantActionBusyId) return;
    setAssistantActionBusyId(suggestion.id);
    setAssistantError("");

    try {
      if (mode === "narrate") {
        const narrationText = String(suggestion?.narrateText || suggestion?.description || "").trim();
        if (!narrationText) throw new Error("No narration text returned for this suggestion.");
        await narrateText(narrationText, suggestion.title || "Session Assistant");
        return;
      }

      const runAction = suggestion?.runAction || {};
      if (runAction.kind === "narrate_scene") {
        await handleNarrateScene(scene);
        return;
      }

      if (runAction.kind === "expand_scene_description") {
        await runSceneBrainAction("expand");
        return;
      }

      if (runAction.kind === "add_scene_twist") {
        await runSceneBrainAction("twist");
        return;
      }

      if (runAction.kind === "start_combat") {
        await handleLaunchEncounter();
        return;
      }

      if (runAction.kind === "npc_whisper") {
        const npc = (campaign?.npcs || []).find((item) => item.name === runAction.npcName);
        if (!npc) throw new Error(`NPC not found for suggestion: ${runAction.npcName || "Unknown NPC"}`);
        await handleNpcWhisper(npc, { busyId: suggestion.id });
        return;
      }

      if (runAction.kind === "npc_dialogue") {
        const npc = (campaign?.npcs || []).find((item) => item.name === runAction.npcName);
        if (!npc) throw new Error(`NPC not found for suggestion: ${runAction.npcName || "Unknown NPC"}`);

        const spokenText = String(runAction.text || "").trim();
        if (!spokenText) throw new Error("No dialogue text returned for this suggestion.");

        const response = await authFetch("/tts/npc-dialogue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ npc_id: String(npc.id || npc.name), text: spokenText }),
        });
        if (!response.ok) {
          throw new Error((await response.text()) || "NPC voice failed.");
        }
        const blob = await response.blob();
        appendSessionEntry("assistant", "npc", `${npc.name}: ${spokenText}`, suggestion.title || "Session Assistant");
        await playAudioBlob(blob);
        return;
      }

      if (runAction.kind === "narrate_text") {
        const narrationText = String(runAction.text || "").trim();
        if (!narrationText) throw new Error("No narration text returned for this suggestion.");
        await narrateText(narrationText, suggestion.title || "Session Assistant");
        return;
      }

      const query = String(runAction.query || suggestion?.description || "").trim();
      if (!query) throw new Error("No rule or lore prompt returned for this suggestion.");
      const response = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) {
        throw new Error((await response.text()) || "Rule explanation failed.");
      }
      const payload = await response.json();
      renderBrainPayload(payload);
    } catch (error) {
      setAssistantError(error?.message || "Session assistant action failed.");
    } finally {
      setAssistantActionBusyId("");
    }
  }, [
    appendSessionEntry,
    assistantActionBusyId,
    authFetch,
    campaign?.npcs,
    handleLaunchEncounter,
    handleNarrateScene,
    handleNpcWhisper,
    narrateText,
    playAudioBlob,
    runSceneBrainAction,
    renderBrainPayload,
    scene,
  ]);

  const handleIgnoreAssistantSuggestion = useCallback((suggestion) => {
    const suggestionId = String(suggestion?.id || "").trim();
    if (!suggestionId) return;
    setIgnoredAssistantSuggestionIds((current) => (
      current.includes(suggestionId) ? current : [...current, suggestionId]
    ));
  }, []);

  useEffect(() => {
    if (!isWakeArmed || isMicActive) {
      stopWakeRecognition();
      return;
    }
    startWakeRecognition();
    return () => {
      stopWakeRecognition();
    };
  }, [isMicActive, isWakeArmed, startWakeRecognition, stopWakeRecognition]);

  useEffect(() => {
    let isActive = true;
    let reconnectAttempt = 0;
    const wsUrl = buildWebSocketUrl("/ws/audio");

    const connect = () => {
      if (!isActive || !wsUrl) return;
      setCoDmStatus("connecting");
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        if (!isActive) return;
        reconnectAttempt = 0;
        setCoDmStatus("open");
        if (isWakeArmedRef.current) {
          setWakeError("");
        }
      };

      ws.onmessage = (event) => {
        if (!isActive) return;
        try {
          const payload = JSON.parse(event.data);
          renderBrainPayload(payload);
        } catch {
          appendActionLog("assistant", String(event.data || ""));
        }
      };

      ws.onerror = () => {
        if (!isActive) return;
        setCoDmStatus("error");
      };

      ws.onclose = () => {
        if (!isActive) return;
        setCoDmStatus("closed");
        if (isWakeArmedRef.current) {
          setWakeError("WebSocket disconnected. Wake listener stays armed but capture will fail until reconnect.");
        }
        stopMicCapture();
        const retryDelay = Math.min(4000, 1000 * (reconnectAttempt + 1));
        reconnectAttempt += 1;
        reconnectTimerRef.current = setTimeout(connect, retryDelay);
      };
    };

    connect();
    return () => {
      isActive = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
      stopMicCapture();
      stopMediaStream();
    };
  }, [appendActionLog, renderBrainPayload, stopMediaStream, stopMicCapture]);

  const submitCoDmQuery = useCallback(async () => {
    const text = coDmQuery.trim();
    if (!text || isSubmittingQuery) return;

    appendActionLog("player", text);
    addSessionLogEntry({ type: "player", text });
    persistSessionEvent(authFetch, { type: "player", text });
    setCoDmQuery("");

    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "query", text }));
      return;
    }

    setIsSubmittingQuery(true);
    try {
      const response = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Failed to query Co-DM.");
      }
      const data = await response.json();
      renderBrainPayload(data);
    } catch (error) {
      appendActionLog("error", error?.message || "Failed to query Co-DM.");
    } finally {
      setIsSubmittingQuery(false);
    }
  }, [appendActionLog, authFetch, coDmQuery, isSubmittingQuery, renderBrainPayload]);

  const toggleWakeArmed = useCallback(() => {
    setWakeError("");
    setIsWakeArmed((current) => {
      const next = !current;
      appendActionLog("assistant", next ? `Wake listener armed for "${WAKE_WORD}".` : "Wake listener disabled.", "Wake");
      return next;
    });
  }, [appendActionLog]);

  const toggleAutoQueryOnVoice = useCallback(() => {
    setAutoQueryOnVoice((current) => !current);
  }, []);

  const assistantSupported = typeof window !== "undefined"
    && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  return (
    <>
      <LiveBoardPage
        campaignData={campaign}
        scene={scene}
        selectedNpcName={selectedNpcName}
        onSelectNpc={setSelectedNpcName}
        onInsertIntoNarration={(text) => setCoDmQuery((prev) => (prev ? prev + "\n" + text : text))}
        authFetch={authFetch}
        onNarrateCampaignAnswer={handleNarrateCampaignAnswer}
        onSceneTrigger={handleSceneTrigger}
        activeSceneTriggerName={activeSceneTriggerName}
        sceneTriggerError={sceneTriggerError}
        sceneSuggestions={sceneSuggestions}
        sceneSuggestionsLoading={sceneSuggestionsLoading}
        sceneSuggestionsError={sceneSuggestionsError}
        onActivateSuggestedScene={handleActivateSuggestedScene}
        activeSuggestedSceneId={activeSuggestedSceneId}
        onLaunchEncounter={handleLaunchEncounter}
        isLaunchingEncounter={isLaunchingEncounter}
        launchEncounterError={launchEncounterError}
        onNarrateScene={handleNarrateScene}
        isNarratingScene={isNarratingScene}
        narrateSceneError={narrateSceneError}
        onExpandSceneDescription={() => runSceneBrainAction("expand")}
        onAddSceneTwist={() => runSceneBrainAction("twist")}
        sceneActionBusy={sceneActionBusy}
        sceneActionError={sceneActionError}
        onSpeakNpcAction={(npc) => openNpcVoiceModal(npc, "speak")}
        onWhisperNpc={handleNpcWhisper}
        assistantSupported={assistantSupported}
        assistantListening={assistantListening}
        assistantAnalyzing={assistantAnalyzing}
        assistantError={assistantError}
        assistantPartialTranscript={assistantPartialTranscript}
        assistantSuggestions={visibleAssistantSuggestions}
        assistantContext={assistantContext}
        actionLog={actionLog}
        assistantActionBusyId={assistantActionBusyId}
        onStartAssistantListening={startAssistantListening}
        onStopAssistantListening={stopAssistantListening}
        onAnalyzeAssistant={analyzeAssistantNow}
        onRunAssistantSuggestion={(suggestion) => handleAssistantSuggestionAction(suggestion, "run")}
        onNarrateAssistantSuggestion={(suggestion) => handleAssistantSuggestionAction(suggestion, "narrate")}
        onIgnoreAssistantSuggestion={handleIgnoreAssistantSuggestion}
        showSessionEmpty={!hasActiveSession}
        onNavigateToPrep={() => onNavigate?.("prep")}
      />
      <NpcVoiceModal
        open={npcVoiceModal.open}
        mode={npcVoiceModal.mode}
        npc={npcVoiceModal.npc}
        value={npcVoiceModal.value}
        onChange={(value) => setNpcVoiceModal((current) => ({ ...current, value }))}
        onClose={closeNpcVoiceModal}
        onSubmit={submitNpcVoiceModal}
        busy={npcVoiceModal.busy}
        error={npcVoiceModal.error}
        generatedText={npcVoiceModal.generatedText}
      />
    </>
  );
};


// ─── App Root ────────────────────────────────────────────────────────────────

function CurrentView() {
  const { campaignData, setCampaignData, authFetch, setBannerState } = useAppState();
  const campaignCtx = useCampaignOptional();
  const activeSessionId = useCampaignContextStore((state) => state.activeSessionId);
  const navigate = useNavigate();
  const location = useLocation();
  const path = (location.pathname || "").replace(/^\/preview\/?/, "") || "/";
  const view = pathToView(path);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [liveCommandActions, setLiveCommandActions] = useState(null);
  const pendingLiveCommandRef = useRef("");
  const campaign = campaignCtx?.campaign ?? campaignData;
  const activeScene = campaignCtx?.activeScene ?? (campaign?.scenes?.[0] || null);
  const onNavigate = useCallback(
    (nextView) => {
      const target = viewToPath[nextView] ?? "/";
      navigate(target);
    },
    [navigate]
  );
  const preferredNpc = resolvePreferredNpc(
    campaign,
    activeScene,
    view === "live" ? liveCommandActions?.preferredNpcName || null : null
  );
  const sceneNarrationText = String(activeScene?.read_aloud || activeScene?.notes || "").trim();

  const handleGuidedSessionStarted = useCallback(
    async (payload) => {
      const campaignId = payload?.campaign_id != null && payload.campaign_id !== ""
        ? String(payload.campaign_id)
        : "";
      let refreshedCampaign = payload?.campaign && typeof payload.campaign === "object" ? payload.campaign : null;

      if (!refreshedCampaign && campaignId) {
        const response = await authFetch(`/api/campaigns/${campaignId}`);
        if (!response.ok) {
          throw new Error((await response.text()) || "Could not load started session.");
        }
        refreshedCampaign = await response.json();
      }
      if (!refreshedCampaign) {
        throw new Error("Could not load started session.");
      }

      const normalized = {
        ...refreshedCampaign,
        party: refreshedCampaign.party ?? [],
        reveals: refreshedCampaign.reveals ?? [],
        items: refreshedCampaign.items ?? [],
        images: refreshedCampaign.images ?? [],
        sessions: Array.isArray(refreshedCampaign.sessions) ? refreshedCampaign.sessions : [],
      };

      setCampaignData(normalized);
      importParseResultToStore(normalized);
      setBackendCampaignId(campaignId || normalized.id);
      onNavigate("live");
    },
    [authFetch, onNavigate, setCampaignData]
  );

  const runLivePaletteCommand = useCallback(
    (commandId) => {
      const handler = liveCommandActions?.[commandId];
      if (view === "live" && typeof handler === "function") {
        void handler();
        return;
      }

      pendingLiveCommandRef.current = commandId;
      if (view !== "live") onNavigate("live");
    },
    [liveCommandActions, onNavigate, view]
  );

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return;
      if (String(event.key || "").toLowerCase() !== "k") return;
      event.preventDefault();
      setCommandPaletteOpen((current) => !current);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    setCommandPaletteOpen(false);
  }, [view]);

  useEffect(() => {
    const pendingCommandId = pendingLiveCommandRef.current;
    if (view !== "live" || !pendingCommandId || !liveCommandActions) return;

    const pendingHandler = liveCommandActions[pendingCommandId];
    if (typeof pendingHandler !== "function") return;

    pendingLiveCommandRef.current = "";
    void pendingHandler();
  }, [liveCommandActions, view]);

  const commandPaletteCommands = [
    {
      id: "narrate-scene",
      title: "Narrate Scene",
      description: activeScene?.title
        ? `Read ${activeScene.title} aloud with the current narration flow.`
        : "Read the active scene aloud with the current narration flow.",
      keywords: ["scene", "narration", "read aloud", "tts", "live"],
      icon: ScrollText,
      group: "Live",
      disabled: !activeSessionId || !sceneNarrationText,
      disabledReason: !activeSessionId
        ? "Start a session first in Guided Session Mode."
        : sceneNarrationText
          ? ""
          : "Add read-aloud text or scene notes to narrate this scene.",
      onSelect: () => runLivePaletteCommand("narrateScene"),
    },
    {
      id: "speak-as-npc",
      title: "Speak as NPC",
      description: preferredNpc
        ? `Open the voice modal for ${preferredNpc.name}.`
        : "Open the active scene NPC voice modal.",
      keywords: ["npc", "voice", "dialogue", "speech", "character"],
      icon: Mic2,
      group: "Live",
      disabled: !activeSessionId || !preferredNpc,
      disabledReason: !activeSessionId
        ? "Start a session first in Guided Session Mode."
        : preferredNpc
          ? ""
          : "Add or select an NPC to use direct voice playback.",
      onSelect: () => runLivePaletteCommand("speakAsNpc"),
    },
    {
      id: "generate-npc-dialogue",
      title: "Generate NPC Dialogue",
      description: preferredNpc
        ? `Open AI dialogue generation for ${preferredNpc.name}.`
        : "Open AI dialogue generation for the current NPC.",
      keywords: ["npc", "ai", "dialogue", "response", "generate"],
      icon: Sparkles,
      group: "Live",
      disabled: !activeSessionId || !preferredNpc,
      disabledReason: !activeSessionId
        ? "Start a session first in Guided Session Mode."
        : preferredNpc
          ? ""
          : "Add or select an NPC before generating dialogue.",
      onSelect: () => runLivePaletteCommand("generateNpcDialogue"),
    },
    {
      id: "open-voice-studio",
      title: "Open Voice Studio",
      description: "Jump to the Voice Studio tool deck.",
      keywords: ["voices", "audio", "narration", "tool deck"],
      icon: Volume2,
      group: "Navigate",
      onSelect: () => onNavigate("voice-studio"),
    },
    {
      id: "open-codex",
      title: "Open Codex",
      description: "Jump to the campaign codex and reference tools.",
      keywords: ["codex", "lore", "search", "reference"],
      icon: BookOpenText,
      group: "Navigate",
      onSelect: () => onNavigate("codex"),
    },
    {
      id: "open-liveboard",
      title: "Open LiveBoard",
      description: "Return to the live session command center.",
      keywords: ["live", "board", "session", "gm control"],
      icon: LayoutDashboard,
      group: "Navigate",
      onSelect: () => onNavigate("live"),
    },
  ];

  let content = null;

  if (view === "prep") {
    content = (
      <PrepPage
        prepContent={
          <PrepRoom view={view} onNavigate={onNavigate} campaignData={campaignData} onUpdateCampaign={setCampaignData} embedded />
        }
        libraryContent={
          <ErrorBoundary>
            <AdventureIntake
              view={view}
              onNavigate={onNavigate}
              campaignData={campaignData}
              onSaveCampaign={setCampaignData}
              authFetch={authFetch}
              embedded
            />
          </ErrorBoundary>
        }
        onNavigate={onNavigate}
      />
    );
  } else if (view === "codex") {
    // Backward-compat: direct /codex URL still works
    content = (
      <ErrorBoundary>
        <CodexPage campaignData={campaignData} authFetch={authFetch} />
      </ErrorBoundary>
    );
  } else if (view === "npc-workshop") {
    // Backward-compat: direct /npcs URL still works
    content = (
      <ErrorBoundary>
        <NPCWorkshopPage campaignData={campaignData} authFetch={authFetch} />
      </ErrorBoundary>
    );
  } else if (view === "voice-studio") {
    content = (
      <ErrorBoundary>
        <VoicePage />
      </ErrorBoundary>
    );
  } else if (view === "campaign") {
    content = (
      <ErrorBoundary>
        <CampaignPage />
      </ErrorBoundary>
    );
  } else if (view === "settings") {
    // Backward-compat: direct /settings URL still works
    content = <SettingsPage />;
  } else {
    content = (
      <ErrorBoundary>
        <LiveBoard
          view={view}
          onNavigate={onNavigate}
          campaignData={campaignData}
          authFetch={authFetch}
          setBannerState={setBannerState}
          defaultAutoQueryOnVoice={true}
          onRegisterCommandActions={setLiveCommandActions}
          onSessionStarted={handleGuidedSessionStarted}
        />
      </ErrorBoundary>
    );
  }

  return (
    <>
      {content}
      <CommandPalette
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        commands={commandPaletteCommands}
      />
    </>
  );
}

export default function App() {
  return (
    <AppStateProvider>
      <CampaignProvider>
        <BrowserRouter basename="/preview">
          <Routes>
            <Route element={<AppShell />}>
              <Route path="*" element={<CurrentView />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </CampaignProvider>
    </AppStateProvider>
  );
}
