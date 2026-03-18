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
import ExtractionReviewQueue from "./components/intake/ExtractionReviewQueue";
import { getPartyPlaceholder, getScenePlaceholder } from "./lib/placeholders";
import AppShell from "./layout/AppShell";
import CommandPalette from "./components/layout/CommandPalette";
import LiveBoardPage from "./app/live-board";
import CodexPage from "./app/codex";
import NPCWorkshopPage from "./app/npcs";
import SettingsPage from "./pages/SettingsPage";
import PrepPage from "./pages/PrepPage";
import VoicePage from "./pages/VoicePage";
import CampaignPage from "./pages/CampaignPage";
import NpcVoiceModal from "./components/live-board/NpcVoiceModal";
import StartSessionPanel from "./components/live-board/StartSessionPanel";
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
  intake: "/intake",
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
const DEFAULT_REVEALS = [
  { name: "Upload adventure docs to see plot hooks", when: "", type: "hook" },
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

// ─── Shared Components ─────────────────────────────────────────────────────

const PrepPanel = ({ title, children, className = "" }) => (
  <section className={`panel-ornate prep-panel ${className}`}>
    <div className="panel-head">
      <div className="plaque">{title}</div>
    </div>
    <div className="panel-body">{children}</div>
  </section>
);

const ViewTabs = ({ view, onNavigate, className = "" }) => (
  <div className={className}>
    <button type="button" className={`nav-glyph-btn ${view === "live" ? "is-active" : ""}`} onClick={() => onNavigate("live")}>
      Live Board
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "codex" ? "is-active" : ""}`} onClick={() => onNavigate("codex")}>
      Campaign Codex
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "npc-workshop" ? "is-active" : ""}`} onClick={() => onNavigate("npc-workshop")}>
      NPC Workshop
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "voice-studio" ? "is-active" : ""}`} onClick={() => onNavigate("voice-studio")}>
      Voice Studio
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "prep" ? "is-active" : ""}`} onClick={() => onNavigate("prep")}>
      Prep Room
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "intake" ? "is-active" : ""}`} onClick={() => onNavigate("intake")}>
      Library
    </button>
    <button type="button" className={`nav-glyph-btn ${view === "settings" ? "is-active" : ""}`} onClick={() => onNavigate("settings")}>
      Settings
    </button>
  </div>
);

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
          if (autoQueryOnVoice) {
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
  }, [appendActionLog, authFetch, autoQueryOnVoice]);

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
        emptyStateContent={
          <StartSessionPanel
            authFetch={authFetch}
            initialCampaign={campaign}
            onSessionStarted={onSessionStarted}
            onOpenLibrary={() => onNavigate?.("intake")}
          />
        }
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


// ─── Prep Room ──────────────────────────────────────────────────────────────

const PrepHeader = ({ view, onNavigate, campaignData }) => (
  <header className="prep-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Prep Room
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">
        {campaignData?.title ? `Active Campaign: ${campaignData.title}` : "Campaign Planning Console"}
      </p>
    </div>
  </header>
);

const BLANK_SCENE = { title: "", act: "", type: "exploration", atmosphere_type: "forest", location: "", read_aloud: "", notes: "", npcs: [] };

const PrepLeftColumn = ({ campaignData, selectedIdx, onSelectScene, onUpdateCampaign }) => {
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const [showForm, setShowForm] = useState(false);
  const [editIdx, setEditIdx] = useState(null); // null = new, number = editing existing
  const [form, setForm] = useState(BLANK_SCENE);

  const existingActs = [...new Set(scenes.map(s => s.act).filter(Boolean))];
  const typeGlyph = { combat: "x", social: "o", exploration: "*", mystery: "?", travel: "~" };

  const actGroups = scenes.reduce((acc, scene, idx) => {
    const act = scene.act || "Adventure";
    if (!acc[act]) acc[act] = [];
    acc[act].push({ ...scene, idx });
    return acc;
  }, {});

  const openNew = () => {
    setForm(BLANK_SCENE);
    setEditIdx(null);
    setShowForm(true);
  };

  const openEdit = (e, scene, idx) => {
    e.stopPropagation();
    setForm({
      title: scene.title || "",
      act: scene.act || "",
      type: scene.type || "exploration",
      atmosphere_type: scene.atmosphere_type || "forest",
      location: scene.location || "",
      read_aloud: scene.read_aloud || "",
      notes: scene.notes || "",
      npcs: scene.npcs || [],
    });
    setEditIdx(idx);
    setShowForm(true);
  };

  const saveScene = () => {
    if (!form.title.trim() || !onUpdateCampaign) return;
    const base = campaignData?.scenes?.length ? campaignData.scenes : [];
    let updated;
    if (editIdx !== null) {
      updated = base.map((s, i) => i === editIdx ? { ...s, ...form } : s);
    } else {
      updated = [...base, { ...form, npcs: [] }];
    }
    onUpdateCampaign({ ...(campaignData || {}), scenes: updated });
    if (editIdx === null) onSelectScene(updated.length - 1);
    setShowForm(false);
    setEditIdx(null);
  };

  return (
    <div className="h-full min-h-0 flex flex-col">
      <PrepPanel title="Adventure Outline" className="flex-1 min-h-0">
        <div className="overflow-y-auto flex-1">
          {Object.entries(actGroups).map(([act, actScenes]) => (
            <div key={act}>
              <div className="prep-act-row"><span>{act}</span><span>▾</span></div>
              {actScenes.map((scene) => (
                <div
                  key={scene.idx}
                  className={`prep-outline-item group flex items-center gap-1 ${scene.idx === selectedIdx ? "is-active" : ""}`}
                  style={{cursor:"pointer"}}
                  onClick={() => onSelectScene(scene.idx)}
                >
                  <div className="prep-outline-glyph">{typeGlyph[scene.type] || "*"}</div>
                  <div className="prep-outline-copy flex-1 min-w-0">
                    <div className="prep-outline-title">{scene.title}</div>
                    <div className="prep-outline-meta">{scene.type}{scene.location ? ` · ${scene.location}` : ""}</div>
                  </div>
                  {onUpdateCampaign && (
                    <button
                      type="button"
                      className="opacity-0 group-hover:opacity-100 text-[#9b7440] hover:text-[#d4af37] text-xs px-1 flex-shrink-0"
                      onClick={(e) => openEdit(e, scene, scene.idx)}
                      title="Edit scene"
                    >✎</button>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>

        {onUpdateCampaign && !showForm && (
          <button type="button" className="nav-glyph-btn intake-parse-btn mt-2 w-full" onClick={openNew}>
            ＋ New Scene
          </button>
        )}

        {showForm && (
          <div className="mt-2 border border-[#4f341f] rounded p-2 space-y-2 bg-[#120a04]">
            <div className="text-xs text-[#d4af37] font-heading mb-1">{editIdx !== null ? "Edit Scene" : "New Scene"}</div>
            <input
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
              placeholder="Title *"
              value={form.title}
              onChange={e => setForm(f => ({...f, title: e.target.value}))}
            />
            <input
              list="act-options"
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
              placeholder="Act (e.g. Chapter 1)"
              value={form.act}
              onChange={e => setForm(f => ({...f, act: e.target.value}))}
            />
            <datalist id="act-options">{existingActs.map(a => <option key={a} value={a} />)}</datalist>
            <div className="flex gap-2">
              <select
                className="flex-1 bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1"
                value={form.type}
                onChange={e => setForm(f => ({...f, type: e.target.value}))}
              >
                {["combat","social","exploration","mystery","travel"].map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              <input
                className="flex-1 bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f]"
                placeholder="Location"
                value={form.location}
                onChange={e => setForm(f => ({...f, location: e.target.value}))}
              />
            </div>
            <select
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1"
              value={form.atmosphere_type || "forest"}
              onChange={e => setForm(f => ({ ...f, atmosphere_type: e.target.value }))}
            >
              {["forest", "tavern", "town", "dungeon", "combat", "mystery"].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <textarea
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f] resize-none"
              placeholder="Read-aloud text..."
              rows={3}
              value={form.read_aloud}
              onChange={e => setForm(f => ({...f, read_aloud: e.target.value}))}
            />
            <textarea
              className="w-full bg-[#1a0f06] border border-[#4f341f] text-[#c8a050] text-xs rounded px-2 py-1 placeholder-[#4f341f] resize-none"
              placeholder="GM notes..."
              rows={2}
              value={form.notes}
              onChange={e => setForm(f => ({...f, notes: e.target.value}))}
            />
            <div className="flex gap-2">
              <button type="button" className="flex-1 nav-glyph-btn intake-parse-btn is-active text-xs py-1" onClick={saveScene} disabled={!form.title.trim()}>
                Save Scene
              </button>
              <button type="button" className="flex-1 nav-glyph-btn intake-parse-btn text-xs py-1" onClick={() => { setShowForm(false); setEditIdx(null); }}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </PrepPanel>
    </div>
  );
};

const PrepMiddleColumn = ({ campaignData, selectedIdx, onUpdateCampaign }) => {
  const [tab, setTab] = useState("readaloud");
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const scene = scenes[selectedIdx] || scenes[0];
  const npcs = campaignData?.npcs?.length ? campaignData.npcs : [];
  const allReveals = campaignData?.reveals?.length ? campaignData.reveals : DEFAULT_REVEALS;
  const sceneNpcs = npcs.filter(n => scene.npcs?.includes(n.name));
  const sceneReveals = (scene.reveals || [])
    .map(name => allReveals.find(r => r.name === name) || { name, type: "clue", when: "" });

  const removeFromScene = (field, value) => {
    if (!onUpdateCampaign || !campaignData) return;
    const updatedScenes = scenes.map((s, i) => {
      if (i !== selectedIdx) return s;
      return { ...s, [field]: (s[field] || []).filter(v => v !== value) };
    });
    onUpdateCampaign({ ...campaignData, scenes: updatedScenes });
  };

  return (
    <div className="h-full min-h-0">
      <PrepPanel title={`Scene: ${scene.title}`} className="h-full">
        {(scene.difficulty || scene.rewards || scene.location) && (
          <div className="flex gap-2 flex-wrap text-xs mb-2">
            {scene.location && <span className="text-[#7a5a30]">📍 {scene.location}</span>}
            {scene.atmosphere_type && <span className="text-[#9b7440]">Atmosphere: {scene.atmosphere_type}</span>}
            {scene.difficulty && scene.difficulty !== "none" && (
              <span className={`border rounded px-2 py-0.5 ${scene.difficulty === "deadly" ? "border-red-900 text-red-400" : scene.difficulty === "hard" ? "border-orange-900 text-orange-400" : "border-[#4f341f] text-[#9b7440]"}`}>{scene.difficulty}</span>
            )}
            {scene.rewards && <span className="text-[#9b7440]">Rewards: {scene.rewards}</span>}
          </div>
        )}
        <div className="tab-strip prep-main-tabs">
          <button type="button" className={tab === "readaloud" ? "tab-active" : ""} onClick={() => setTab("readaloud")}>
            Read-aloud
          </button>
          <button type="button" className={tab === "npcs" ? "tab-active" : ""} onClick={() => setTab("npcs")}>
            Important NPCs
          </button>
          <button type="button" className={tab === "secrets" ? "tab-active" : ""} onClick={() => setTab("secrets")}>
            Secrets &amp; Clues
          </button>
          <button type="button" className={tab === "notes" ? "tab-active" : ""} onClick={() => setTab("notes")}>
            GM Notes
          </button>
        </div>

        {tab === "readaloud" && (
          <div className="parchment prep-readaloud">
            {scene.read_aloud || "No read-aloud text extracted for this scene."}
          </div>
        )}

        {tab === "npcs" && (
          <div className="prep-npc-grid mt-2">
            {sceneNpcs.length ? sceneNpcs.map((npc) => (
              <article key={npc.name} className="prep-npc-card" style={{ position:"relative" }}>
                {onUpdateCampaign && (
                  <button
                    type="button"
                    onClick={() => removeFromScene("npcs", npc.name)}
                    title="Remove from scene"
                    style={{ position:"absolute", top:"2px", right:"2px", background:"none",
                      border:"none", color:"#7a3a3a", fontSize:"0.8rem", cursor:"pointer",
                      lineHeight:1, padding:"0 2px" }}
                  >×</button>
                )}
                <div className="prep-npc-face">{npc.name.slice(0, 2).toUpperCase()}</div>
                <p>{npc.name}</p>
                <p className="text-xs text-[#9b7440]">{npc.role}</p>
              </article>
            )) : (
              <div className="intake-empty">No NPCs assigned. Use Library Assets → +</div>
            )}
          </div>
        )}

        {tab === "secrets" && (
          <div className="prep-reveal-list mt-2">
            {sceneReveals.length ? sceneReveals.map((reveal) => (
              <div key={reveal.name} className="prep-reveal-row" style={{ display:"flex", alignItems:"center" }}>
                <span className={`prep-reveal-dot ${reveal.type === "hook" ? "green" : reveal.type === "secret" ? "red" : "amber"}`} />
                <span className="prep-reveal-name" style={{ flex:1 }}>{reveal.name}</span>
                <span className="prep-reveal-status">{reveal.when || ""}</span>
                {onUpdateCampaign && (
                  <button
                    type="button"
                    onClick={() => removeFromScene("reveals", reveal.name)}
                    title="Remove from scene"
                    style={{ background:"none", border:"none", color:"#7a3a3a",
                      fontSize:"0.8rem", cursor:"pointer", lineHeight:1,
                      marginLeft:"0.25rem", padding:"0 2px" }}
                  >×</button>
                )}
              </div>
            )) : (
              <div className="intake-empty">No secrets assigned. Use Library Assets → +</div>
            )}
          </div>
        )}

        {tab === "notes" && (
          <div className="parchment prep-readaloud mt-2">
            {scene.notes || "No GM notes for this scene."}
          </div>
        )}
      </PrepPanel>
    </div>
  );
};

const PrepRightColumn = ({ campaignData, selectedIdx, onUpdateCampaign }) => {
  const allNpcs = campaignData?.npcs || [];
  const allReveals = campaignData?.reveals || [];
  const allItems = campaignData?.items || [];
  const scenes = campaignData?.scenes?.length ? campaignData.scenes : DEFAULT_SCENES;
  const [assetTab, setAssetTab] = useState("npcs");

  const addToScene = (field, value) => {
    if (!onUpdateCampaign || !campaignData) return;
    const updatedScenes = scenes.map((s, i) => {
      if (i !== selectedIdx) return s;
      const arr = s[field] || [];
      if (arr.includes(value)) return s;
      return { ...s, [field]: [...arr, value] };
    });
    onUpdateCampaign({ ...campaignData, scenes: updatedScenes });
  };

  const hasAny = allNpcs.length || allReveals.length || allItems.length;

  const AssetRow = ({ label, field }) => (
    <div style={{ display:"flex", alignItems:"center", gap:"0.4rem", padding:"0.3rem 0.25rem",
      borderBottom:"1px solid #2e1e0a" }}>
      <span style={{ flex:1, fontSize:"0.8rem", color:"#c9a85c", fontFamily:"Cinzel,serif",
        overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{label}</span>
      <button
        type="button"
        onClick={() => addToScene(field, label)}
        title="Add to scene"
        style={{ background:"none", border:"1px solid #5a3e1b", color:"#d4af37",
          borderRadius:"3px", width:"18px", height:"18px", lineHeight:"14px",
          fontSize:"0.9rem", cursor:"pointer", flexShrink:0, textAlign:"center" }}
      >+</button>
    </div>
  );

  return (
    <div className="h-full min-h-0">
      <PrepPanel title="Library Assets" className="h-full">
        {hasAny ? (
          <>
            <div className="tab-strip prep-main-tabs">
              <button type="button" className={assetTab === "npcs" ? "tab-active" : ""} onClick={() => setAssetTab("npcs")}>All NPCs</button>
              <button type="button" className={assetTab === "secrets" ? "tab-active" : ""} onClick={() => setAssetTab("secrets")}>All Secrets</button>
              <button type="button" className={assetTab === "items" ? "tab-active" : ""} onClick={() => setAssetTab("items")}>All Items</button>
            </div>
            <div style={{ overflowY:"auto", flex:1, marginTop:"0.25rem" }}>
              {assetTab === "npcs" && (
                allNpcs.length
                  ? allNpcs.map(n => <AssetRow key={n.name} label={n.name} field="npcs" />)
                  : <div className="intake-empty">No NPCs in campaign.</div>
              )}
              {assetTab === "secrets" && (
                allReveals.length
                  ? allReveals.map(r => <AssetRow key={r.name} label={r.name} field="reveals" />)
                  : <div className="intake-empty">No secrets in campaign.</div>
              )}
              {assetTab === "items" && (
                allItems.length
                  ? allItems.map(it => <AssetRow key={it.name} label={it.name} field="items" />)
                  : <div className="intake-empty">No items in campaign.</div>
              )}
            </div>
          </>
        ) : (
          <div className="intake-empty">Use Library to import a campaign.</div>
        )}
      </PrepPanel>
    </div>
  );
};

const PrepRoom = ({ view, onNavigate, campaignData, onUpdateCampaign }) => {
  const [selectedIdx, setSelectedIdx] = useState(0);
  return (
    <div className="dm-shell dm-fit prep-shell mx-auto">
      <PrepHeader view={view} onNavigate={onNavigate} campaignData={campaignData} />
      <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-3 min-h-0">
          <PrepLeftColumn campaignData={campaignData} selectedIdx={selectedIdx} onSelectScene={setSelectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
        <div className="xl:col-span-5 min-h-0">
          <PrepMiddleColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
        <div className="xl:col-span-4 min-h-0">
          <PrepRightColumn campaignData={campaignData} selectedIdx={selectedIdx} onUpdateCampaign={onUpdateCampaign} />
        </div>
      </section>
    </div>
  );
};

// ─── Library ─────────────────────────────────────────────────────────────

const IntakeHeader = ({ view, onNavigate, campaignData }) => (
  <header className="prep-header intake-header">
    <div className="header-glow" />
    <ViewTabs view={view} onNavigate={onNavigate} className="prep-header-bar nav-tab-bar" />
    <div className="relative z-10 text-center">
      <h1 className="font-heading text-[clamp(1.8rem,2.35vw,3rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
        GM Voice Studio - Library
      </h1>
      <p className="font-heading text-[clamp(1rem,1.5vw,1.85rem)] leading-[1.05] text-[#d8b36f]">
        {campaignData?.title ? `Active Campaign: ${campaignData.title}` : "Upload docs · AI extracts · Feed your session"}
      </p>
    </div>
  </header>
);

const TYPE_BADGE = { hook: "green", secret: "red", clue: "amber", twist: "amber" };

const DetailDrawer = ({ item, onClose, onLightbox }) => {
  if (!item) return null;
  const { type, data } = item;
  return (
    <div className="fixed inset-0 z-40 flex justify-end" onClick={onClose} style={{ backdropFilter:"blur(4px)", WebkitBackdropFilter:"blur(4px)" }}>
      <div className="w-full max-w-sm h-full bg-[#1a0f06] border-l-2 border-[#4f341f] shadow-2xl flex flex-col overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-3 py-2 border-b border-[#4f341f] bg-[#120a04]">
          <div className="font-heading text-[#d4af37] text-sm truncate">
            {type === "scene" && `Scene: ${data.title}`}
            {type === "npc" && `NPC: ${data.name}`}
            {type === "location" && `Location: ${data.name}`}
            {type === "reveal" && `${data.type || "Reveal"}: ${data.name}`}
            {type === "item" && `Item: ${data.name}`}
          </div>
          <button type="button" onClick={onClose} className="text-[#9b7440] hover:text-[#d4af37] text-2xl ml-2 flex-shrink-0 leading-none">×</button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {type === "scene" && <>
            <div className="flex gap-2 flex-wrap text-xs">
              {data.act && <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.act}</span>}
              {data.type && <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.type}</span>}
              {data.difficulty && data.difficulty !== "none" && (
                <span className={`border rounded px-2 py-0.5 ${data.difficulty === "deadly" ? "border-red-900 text-red-400" : data.difficulty === "hard" ? "border-orange-900 text-orange-400" : "border-[#4f341f] text-[#9b7440]"}`}>{data.difficulty}</span>
              )}
            </div>
            {data.location && <p className="text-xs text-[#7a5a30]">Location: {data.location}</p>}
            {data.image_url && <img src={data.image_url} alt={data.title} className="w-full rounded border border-[#4f341f] cursor-pointer object-cover" style={{maxHeight:"140px"}} onClick={() => onLightbox(data.image_url)} />}
            {data.read_aloud && <><div className="subhead">Read-Aloud</div><div className="parchment text-sm">{data.read_aloud}</div></>}
            {data.npcs?.length > 0 && <><div className="subhead">NPCs in Scene</div><div className="flex flex-wrap gap-1">{data.npcs.map(n => <span key={n} className="border border-[#4f341f] rounded px-2 py-0.5 text-xs text-[#9b7440]">{n}</span>)}</div></>}
            {data.rewards && <><div className="subhead">Rewards</div><p className="text-xs text-[#9b7440]">{data.rewards}</p></>}
            {data.notes && <><div className="subhead">GM Notes</div><p className="text-xs text-[#7a5a30] italic">{data.notes}</p></>}
          </>}

          {type === "npc" && <>
            <div className="flex gap-3 items-start">
              {data.image_url
                ? <img src={data.image_url} alt={data.name} className="w-20 h-20 object-cover rounded border border-[#4f341f] flex-shrink-0 cursor-pointer" onClick={() => onLightbox(data.image_url)} />
                : <div className="w-20 h-20 flex-shrink-0 rounded border border-[#4f341f] bg-[#120a04] flex items-center justify-center text-[#4f341f] text-2xl font-bold">{(data.name || "?").slice(0,2).toUpperCase()}</div>
              }
              <div>
                <div className="flex gap-2 flex-wrap text-xs mb-1">
                  <span className="border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.role}</span>
                  {data.faction && <span className="text-[#7a5a30]">{data.faction}</span>}
                </div>
                <div className="flex gap-3 text-xs text-[#c8a050]">
                  {data.hp && <span>HP {data.hp}</span>}
                  {data.ac ? <span>AC {data.ac}</span> : null}
                  {data.cr && <span>{data.cr}</span>}
                </div>
              </div>
            </div>
            {data.personality && <><div className="subhead">Personality</div><p className="text-xs text-[#9b7440] italic">{data.personality}</p></>}
            {data.motivation && <><div className="subhead">Motivation</div><p className="text-xs text-[#b08040]">{data.motivation}</p></>}
            {data.secrets && <><div className="subhead">Secrets</div><p className="text-xs text-[#7a5a30] italic">{data.secrets}</p></>}
          </>}

          {type === "location" && <>
            {data.scene && <p className="text-xs text-[#7a5a30]">Scene: {data.scene}</p>}
            <div className="subhead">Description</div>
            <p className="text-sm text-[#c8a050]">{data.description || "No description available."}</p>
          </>}

          {type === "reveal" && <>
            <div className="flex gap-2 items-center mb-1">
              <span className={`prep-reveal-dot ${TYPE_BADGE[data.type] || "amber"}`} />
              <span className="text-xs border border-[#4f341f] rounded px-2 py-0.5 text-[#9b7440]">{data.type}</span>
            </div>
            {data.when && <><div className="subhead">When Triggered</div><p className="text-sm text-[#c8a050]">{data.when}</p></>}
          </>}

          {type === "item" && <>
            {data.magical && <span className="text-xs border border-amber-800 rounded px-2 py-0.5 text-amber-400">Magical</span>}
            {data.scene && <p className="text-xs text-[#7a5a30] mt-1">Found in: {data.scene}</p>}
            <div className="subhead">Description</div>
            <p className="text-sm text-[#c8a050]">{data.description || "No description."}</p>
          </>}
        </div>
      </div>
    </div>
  );
};

// Review tab button — shows a live count badge from the review queue store.
const ReviewTabButton = ({ activePanel, setActivePanel }) => {
  const items = useExtractionReviewQueueStore((s) => s.items);
  const pendingCount = items.filter(
    (i) => i.reviewStatus === "pending" || i.reviewStatus === "needs_review"
  ).length;
  return (
    <button
      type="button"
      className={activePanel === "review" ? "tab-active" : ""}
      onClick={() => setActivePanel("review")}
    >
      Review{items.length > 0 ? ` (${items.length}${pendingCount > 0 ? ` · ${pendingCount} pending` : ""})` : ""}
    </button>
  );
};

const AdventureIntake = ({ view, onNavigate, campaignData, onSaveCampaign, authFetch }) => {
  const { clearCampaignData } = useAppState();
  const { enqueueBatch } = useExtractionReviewQueueStore();
  const [files, setFiles] = useState([]);
  const [isParsing, setIsParsing] = useState(false);
  const [isExtractingImages, setIsExtractingImages] = useState(false);
  const [parseError, setParseError] = useState("");
  const [parseResult, setParseResult] = useState(() => {
    try {
      const s = localStorage.getItem("gm_parse_result");
      if (!s) return null;
      const parsed = JSON.parse(s);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch { return null; }
  });
  const [images, setImages] = useState(() => {
    try {
      const s = localStorage.getItem("gm_parse_images");
      if (!s) return { embedded: [], pages: [] };
      const parsed = JSON.parse(s);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed) &&
          Array.isArray(parsed.embedded) && Array.isArray(parsed.pages)) {
        return parsed;
      }
      return { embedded: [], pages: [] };
    } catch { return { embedded: [], pages: [] }; }
  });
  const [saved, setSaved] = useState(false);
  const [activePanel, setActivePanel] = useState("outline");
  const [lightbox, setLightbox] = useState(null); // URL of enlarged image
  const [detailItem, setDetailItem] = useState(null); // {type, data} for detail drawer
  const [expandedActs, setExpandedActs] = useState(new Set()); // which chapter headers are open
  const [campaignSystems, setCampaignSystems] = useState(() => listGameSystemPlugins());
  const [selectedSystemId, setSelectedSystemId] = useState(DEFAULT_GAME_SYSTEM_ID);
  const selectedSystem = useMemo(
    () => resolveGameSystemPlugin(selectedSystemId, campaignSystems),
    [selectedSystemId, campaignSystems]
  );

  const deleteAssignedImage = (idx) =>
    setParseResult(r => ({ ...r, images: r.images.filter(img => img.idx !== idx) }));
  const deleteEmbeddedImage = (i) =>
    setImages(s => ({ ...s, embedded: s.embedded.filter((_, j) => j !== i) }));
  const deletePageImage = (i) =>
    setImages(s => ({ ...s, pages: s.pages.filter((_, j) => j !== i) }));
  const assignImageTo = (idx, entityName) =>
    setParseResult(r => {
      const targetImg = r.images.find(img => img.idx === idx);
      const url = targetImg?.url || null;
      const updatedImages = r.images.map(img =>
        img.idx === idx ? { ...img, assigned_to: entityName } : img
      );
      const updatedNpcs = (r.npcs || []).map(n => {
        if (entityName && n.name === entityName) return { ...n, image_url: url };
        if (!entityName && n.image_url === url) return { ...n, image_url: null };
        return n;
      });
      const updatedScenes = (r.scenes || []).map(s => {
        if (entityName && s.title === entityName) return { ...s, image_url: url };
        if (!entityName && s.image_url === url) return { ...s, image_url: null };
        return s;
      });
      return { ...r, images: updatedImages, npcs: updatedNpcs, scenes: updatedScenes };
    });

  const handleAssignImage = (imageUrl, selection) =>
    setParseResult(r => {
      if (!r) return r;
      const [type, idxStr] = (selection || "").split(":");
      const i = parseInt(idxStr, 10);
      const updatedNpcs = (r.npcs || []).map((n, ni) =>
        type === "npc" && ni === i ? { ...n, image_url: imageUrl }
        : n.image_url === imageUrl ? { ...n, image_url: null } : n
      );
      const updatedScenes = (r.scenes || []).map((s, si) =>
        type === "scene" && si === i ? { ...s, image_url: imageUrl }
        : s.image_url === imageUrl ? { ...s, image_url: null } : s
      );
      return { ...r, npcs: updatedNpcs, scenes: updatedScenes };
    });

  const getAssignedLabel = (url) => {
    if (!parseResult) return null;
    const npc = (parseResult.npcs || []).find(n => n.image_url === url);
    if (npc) return `NPC: ${npc.name}`;
    const scene = (parseResult.scenes || []).find(s => s.image_url === url);
    if (scene) return `Scene: ${scene.title}`;
    return null;
  };

  const [savedCampaigns, setSavedCampaigns] = useState([]);
  const [loadingCampaigns, setLoadingCampaigns] = useState(false);
  const [loadingCampaignId, setLoadingCampaignId] = useState(null);

  useEffect(() => {
    try {
      if (parseResult) localStorage.setItem("gm_parse_result", JSON.stringify(parseResult));
      else localStorage.removeItem("gm_parse_result");
    } catch {}
  }, [parseResult]);

  useEffect(() => {
    try {
      if (images.embedded.length || images.pages.length)
        localStorage.setItem("gm_parse_images", JSON.stringify(images));
      else localStorage.removeItem("gm_parse_images");
    } catch {}
  }, [images]);

  useEffect(() => {
    setLoadingCampaigns(true);
    authFetch("/api/campaigns")
      .then(r => r.ok ? r.json() : [])
      .then(data => setSavedCampaigns(Array.isArray(data) ? data : []))
      .catch(() => setSavedCampaigns([]))
      .finally(() => setLoadingCampaigns(false));
  }, [authFetch]);

  useEffect(() => {
    let cancelled = false;
    authFetch("/api/campaign-systems")
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (cancelled || !payload) return;
        const nextSystems = Array.isArray(payload.systems)
          ? payload.systems
            .map((system) => normalizeGameSystemPlugin(system))
            .filter(Boolean)
          : [];
        if (nextSystems.length > 0) {
          setCampaignSystems(nextSystems);
        }
        const defaultSystemId = normalizeGameSystemId(payload.default_system_id);
        setSelectedSystemId((current) => normalizeGameSystemId(current || defaultSystemId));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [authFetch]);

  useEffect(() => {
    const parsedSystemId = parseResult?.system_id ?? parseResult?.systemId;
    if (parsedSystemId) {
      setSelectedSystemId(normalizeGameSystemId(parsedSystemId));
    }
  }, [parseResult?.system_id, parseResult?.systemId]);

  const onFileChange = (e) => {
    setFiles(Array.from(e.target.files || []));
    setParseError("");
    setSaved(false);
  };

  const runParse = async (endpoint) => {
    if (!files.length) {
      setParseError("Select at least one .txt, .md, or .pdf file.");
      return;
    }
    setParseError("");
    setIsParsing(true);
    setParseResult(null);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      formData.append("campaign_system", selectedSystemId);
      const res = await authFetch(endpoint, { method: "POST", body: formData });
      const raw = await res.text();
      let payload = null;
      try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
      if (!res.ok) throw new Error((payload?.detail) || raw || `Parse failed (${res.status})`);
      if (!payload) throw new Error("Parse returned no data.");
      setParseResult(payload);
      // Persist backend campaign ID for sync operations
      if (payload.campaign_id) setBackendCampaignId(payload.campaign_id);
      // Enqueue extracted entities into review queue
      try {
        const batch = parseResultToExtractionBatch(
          payload,
          files.length === 1 ? files[0].name : (payload.title || undefined)
        );
        if (batch.entities.length > 0) {
          enqueueBatch(batch);
          setActivePanel("review");
        }
      } catch { /* non-fatal — review queue is optional */ }
      // Refresh saved campaigns list (new campaign was just persisted to DB)
      authFetch("/api/campaigns")
        .then(r => r.ok ? r.json() : [])
        .then(data => setSavedCampaigns(Array.isArray(data) ? data : []))
        .catch(() => {});
      // Auto-expand the first act/chapter in the outline
      const firstAct = payload?.scenes?.[0]?.act || (payload?.acts?.[0]?.title);
      if (firstAct) setExpandedActs(new Set([firstAct]));
      // Auto-switch to images tab if AI parse returned images
      if (payload?.images?.length > 0) {
        setActivePanel("images");
      }
    } catch (err) {
      setParseError(err.message || "Unable to parse documents.");
    } finally {
      setIsParsing(false);
    }
  };

  const runImageExtract = async () => {
    if (!files.length) {
      setParseError("Select at least one PDF file first.");
      return;
    }
    setParseError("");
    setIsExtractingImages(true);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      const res = await authFetch("/adventure/images", { method: "POST", body: formData });
      const raw = await res.text();
      let payload = null;
      try { payload = raw ? JSON.parse(raw) : null; } catch { payload = null; }
      if (!res.ok) throw new Error((payload?.detail) || `Image extraction failed (${res.status})`);
      setImages({ embedded: payload.embedded || [], pages: payload.pages || [] });
      setActivePanel("images");
    } catch (err) {
      setParseError(err.message || "Image extraction failed.");
    } finally {
      setIsExtractingImages(false);
    }
  };

  const saveToSession = () => {
    if (!parseResult) return;
    onSaveCampaign(parseResult);           // legacy AppState + localStorage (unchanged)
    try { importParseResultToStore(parseResult); } catch { /* non-fatal */ }
    setSaved(true);
  };

  const loadSavedCampaign = async (id) => {
    setLoadingCampaignId(id);
    try {
      const res = await authFetch(`/api/campaigns/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      const normalized = {
        ...data,
        party: data.party ?? [],
        reveals: data.reveals ?? [],
        items: data.items ?? [],
        images: data.images ?? [],
      };
      setParseResult(normalized);
      onSaveCampaign(normalized);          // sync to AppState so legacy views update
      try { importParseResultToStore(normalized); } catch { /* non-fatal */ }
      setBackendCampaignId(id);            // persist backend ID for sync operations
      setSaved(true);
      setActivePanel("outline");
      const firstAct = data?.scenes?.[0]?.act;
      if (firstAct) setExpandedActs(new Set([firstAct]));
    } finally {
      setLoadingCampaignId(null);
    }
  };

  const deleteSavedCampaign = async (id) => {
    if (!window.confirm("Delete this campaign? This cannot be undone.")) return;
    await authFetch(`/api/campaigns/${id}`, { method: "DELETE" });
    setSavedCampaigns(prev => prev.filter(c => c.id !== id));
    if (parseResult?.id === id) { setParseResult(null); setSaved(false); }
  };

  const clearAllCampaignData = async () => {
    if (!window.confirm("Clear all campaign data? This will remove your current parse, any loaded campaign, and delete all saved campaigns from the server. You can then upload new documents. This cannot be undone.")) return;
    await clearCampaignData({ deleteBackendCampaigns: true });
    setParseResult(null);
    setSaved(false);
    setFiles([]);
    setImages({ embedded: [], pages: [] });
    setSavedCampaigns([]);
    setParseError("");
    setActivePanel("outline");
    setDetailItem(null);
    setLightbox(null);
    setExpandedActs(new Set());
    setSelectedSystemId(DEFAULT_GAME_SYSTEM_ID);
  };

  const npcs = parseResult?.npcs?.length ? parseResult.npcs : [];
  const party = parseResult?.party?.length ? parseResult.party : [];
  const scenes = parseResult?.scenes?.length ? parseResult.scenes : [];
  const locations = parseResult?.locations?.length ? parseResult.locations : [];
  const reveals = parseResult?.reveals?.length ? parseResult.reveals : [];
  const items = parseResult?.items?.length ? parseResult.items : [];

  // Quick parse result (regex-based) also has acts/npcs as strings
  const acts = parseResult?.acts || [];
  const isAiResult = parseResult && typeof parseResult.npcs?.[0] === "object";

  return (
    <div className="dm-shell dm-fit prep-shell intake-shell mx-auto">
      <IntakeHeader view={view} onNavigate={onNavigate} campaignData={campaignData} />
      <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">

        {/* Left: Upload */}
        <div className="xl:col-span-3 min-h-0">
          <PrepPanel title="Upload Adventure Docs" className="h-full">
            {/* Saved Campaigns */}
            <div style={{ marginBottom: "1rem" }}>
              <h3 style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.85rem",
                letterSpacing: "0.05em", marginBottom: "0.5rem", borderBottom: "1px solid #5a3e1b",
                paddingBottom: "0.35rem" }}>
                Saved Campaigns
              </h3>
              {loadingCampaigns ? (
                <p style={{ color: "#9c7a3a", fontSize: "0.78rem" }}>Loading…</p>
              ) : (
                <>
                  <label style={{ display:"block", color:"#9c7a3a", fontSize:"0.72rem",
                    fontFamily:"Cinzel,serif", letterSpacing:"0.04em", marginBottom:"0.3rem" }}>
                    Active Campaign:
                  </label>
                  <div style={{ display:"flex", gap:"0.4rem", alignItems:"center" }}>
                    <select
                      value={parseResult?.id ?? ""}
                      onChange={e => { if (e.target.value) loadSavedCampaign(Number(e.target.value)); }}
                      style={{ flex:1, background:"#1a0f06", color:"#e7c27a",
                        border:"1px solid #5a3e1b", borderRadius:"4px",
                        padding:"0.4rem 0.5rem", fontFamily:"Cinzel,serif", fontSize:"0.8rem",
                        cursor:"pointer" }}
                    >
                      <option value="" style={{ color:"#6b5230" }}>— Select Campaign —</option>
                      {savedCampaigns.map(c => (
                        <option key={c.id} value={c.id}>
                          {c.title || `Campaign #${c.id}`}{c.system_label ? ` · ${c.system_label}` : ""}
                        </option>
                      ))}
                    </select>
                    {parseResult?.id && (
                      <button
                        onClick={() => deleteSavedCampaign(parseResult.id)}
                        title="Delete campaign"
                        style={{ background:"none", border:"1px solid #7a2020", borderRadius:"4px",
                          color:"#c05050", padding:"0.35rem 0.5rem", cursor:"pointer", flexShrink:0,
                          display:"flex", alignItems:"center" }}
                        onMouseEnter={e => { e.currentTarget.style.background="#3a0f0f"; e.currentTarget.style.color="#ff6b6b"; }}
                        onMouseLeave={e => { e.currentTarget.style.background="none"; e.currentTarget.style.color="#c05050"; }}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                  <div style={{ marginTop: "0.6rem" }}>
                    <button
                      type="button"
                      onClick={clearAllCampaignData}
                      title="Clear parse result, loaded campaign, and all saved campaigns so you can upload new documents"
                      style={{ background: "none", border: "1px solid #5a3e1b", borderRadius: "4px",
                        color: "#9c7a3a", padding: "0.35rem 0.6rem", cursor: "pointer", fontSize: "0.75rem",
                        fontFamily: "Cinzel, serif" }}
                      onMouseEnter={e => { e.currentTarget.style.background = "#2a1f10"; e.currentTarget.style.color = "#c8a050"; }}
                      onMouseLeave={e => { e.currentTarget.style.background = "none"; e.currentTarget.style.color = "#9c7a3a"; }}
                    >
                      Clear all campaign data (new upload)
                    </button>
                  </div>
                  {loadingCampaignId && (
                    <p style={{ color:"#9c7a3a", fontSize:"0.72rem", marginTop:"0.3rem" }}>Loading…</p>
                  )}
                </>
              )}
            </div>

            <p className="intake-hint">
              Drop in session notes, module PDFs, or campaign text. AI Parse uses Claude to extract full campaign data.
            </p>

            <div style={{ marginBottom: "0.9rem" }}>
              <h3 style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.85rem",
                letterSpacing: "0.05em", marginBottom: "0.5rem", borderBottom: "1px solid #5a3e1b",
                paddingBottom: "0.35rem" }}>
                Campaign System
              </h3>
              <label style={{ display:"block", color:"#9c7a3a", fontSize:"0.72rem",
                fontFamily:"Cinzel,serif", letterSpacing:"0.04em", marginBottom:"0.3rem" }}>
                Rules Preset:
              </label>
              <select
                value={selectedSystemId}
                onChange={(e) => setSelectedSystemId(normalizeGameSystemId(e.target.value))}
                style={{ width:"100%", background:"#1a0f06", color:"#e7c27a",
                  border:"1px solid #5a3e1b", borderRadius:"4px", padding:"0.45rem 0.55rem",
                  fontFamily:"Cinzel,serif", fontSize:"0.82rem", cursor:"pointer" }}
              >
                {campaignSystems.map((system) => (
                  <option key={system.id} value={system.id}>{system.label}</option>
                ))}
              </select>
              {selectedSystem && (
                <div style={{ marginTop: "0.55rem", border: "1px solid #4f341f", borderRadius: "6px",
                  background: "rgba(32, 18, 8, 0.72)", padding: "0.65rem 0.75rem" }}>
                  <div style={{ color: "#e7c27a", fontFamily: "Cinzel, serif", fontSize: "0.84rem", marginBottom: "0.35rem" }}>
                    {selectedSystem.label}
                  </div>
                  <p style={{ color: "#b89a62", fontSize: "0.74rem", lineHeight: 1.5, marginBottom: "0.35rem" }}>
                    {selectedSystem.rules_flavor}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45, marginBottom: "0.25rem" }}>
                    Terms: {selectedSystem.skill_check_terminology.skill_term}, {selectedSystem.skill_check_terminology.check_term}, {selectedSystem.skill_check_terminology.difficulty_term}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45, marginBottom: "0.25rem" }}>
                    Encounters: {selectedSystem.encounter_assumptions}
                  </p>
                  <p style={{ color: "#9c7a3a", fontSize: "0.72rem", lineHeight: 1.45 }}>
                    Tone: {selectedSystem.thematic_guidance}
                  </p>
                </div>
              )}
            </div>

            <div
              style={{ border:"2px dashed #4f341f", borderRadius:"6px",
                padding:"0.75rem", marginBottom:"0.5rem", transition:"border-color 0.2s" }}
              onMouseEnter={e => e.currentTarget.style.borderColor="#9b7440"}
              onMouseLeave={e => e.currentTarget.style.borderColor="#4f341f"}
            >
              <label className="intake-file-pick">
                <Upload size={16} className="inline mr-1" />
                <span>Select Files</span>
                <input type="file" multiple accept=".txt,.md,.pdf" onChange={onFileChange} />
              </label>
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn is-active"
                onClick={() => runParse("/adventure/ai-parse")}
                disabled={isParsing || isExtractingImages}
              >
                <Zap size={14} className="inline mr-1" />
                {isParsing ? "Parsing with AI..." : "AI Parse (Claude)"}
              </button>
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn"
                onClick={() => runParse("/adventure/parse")}
                disabled={isParsing || isExtractingImages}
              >
                {isParsing ? "Parsing..." : "Quick Parse (fast)"}
              </button>
              <button
                type="button"
                className="nav-glyph-btn intake-parse-btn"
                onClick={runImageExtract}
                disabled={isParsing || isExtractingImages}
              >
                <Map size={14} className="inline mr-1" />
                {isExtractingImages ? "Extracting images..." : "Extract Images"}
              </button>
            </div>

            {parseError && <div className="intake-error mt-2">{parseError}</div>}

            <div className="subhead mt-3">Queued Files</div>
            <div className="intake-file-list">
              {files.length ? files.map((f) => (
                <div key={`${f.name}-${f.size}`} className="intake-file-item">
                  <span>{f.name}</span>
                  <small>{Math.max(1, Math.round(f.size / 1024))} KB</small>
                </div>
              )) : (
                <div className="intake-empty">No files selected.</div>
              )}
            </div>

            {parseResult && (
              <button
                type="button"
                className={`nav-glyph-btn intake-parse-btn mt-4 ${saved ? "is-active" : ""}`}
                onClick={saveToSession}
              >
                {saved
                  ? <><CheckCircle size={14} className="inline mr-1" />Saved to Campaign</>
                  : <><Save size={14} className="inline mr-1" />Save to Campaign</>
                }
              </button>
            )}

            {parseResult?.files?.length && (
              <div className="mt-3 text-xs text-[#7a5a30]">
                {parseResult.files.map(f => (
                  <div key={f.name}>{f.name} — {(f.characters / 1000).toFixed(1)}k chars{f.page_count ? ` · ${f.page_count}p` : ""}</div>
                ))}
              </div>
            )}
          </PrepPanel>
        </div>

        {/* Middle: Outline / Summary / Images */}
        <div className="xl:col-span-4 min-h-0">
          <PrepPanel title="Parsed Adventure Outline" className="h-full">
            <div className="tab-strip mb-2">
              <button type="button" className={activePanel === "outline" ? "tab-active" : ""} onClick={() => setActivePanel("outline")}>
                Outline
              </button>
              <button type="button" className={activePanel === "locations" ? "tab-active" : ""} onClick={() => setActivePanel("locations")}>
                Locations
              </button>
              <button type="button" className={activePanel === "party" ? "tab-active" : ""} onClick={() => setActivePanel("party")}>
                Party
              </button>
              <button type="button" className={activePanel === "items" ? "tab-active" : ""} onClick={() => setActivePanel("items")}>
                Items {items.length > 0 ? `(${items.length})` : ""}
              </button>
              <button type="button" className={activePanel === "images" ? "tab-active" : ""} onClick={() => setActivePanel("images")}>
                Images {(images.embedded.length + images.pages.length) > 0 ? `(${images.embedded.length + images.pages.length})` : ""}
              </button>
              <ReviewTabButton activePanel={activePanel} setActivePanel={setActivePanel} />
            </div>

            {parseResult?.summary && activePanel !== "images" && (
              <div className="parchment intake-summary mb-3">
                {parseResult.summary}
              </div>
            )}

            {!parseResult && activePanel !== "images" && (
              <div className="intake-empty">Upload docs and click a Parse button to see your adventure outline here.</div>
            )}

            {/* Images gallery panel */}
            {activePanel === "images" && (
              <div className="overflow-y-auto space-y-3">
                {/* AI-assigned images from ai-parse (rich metadata) */}
                {parseResult?.images?.length > 0 ? (
                  <>
                    <div className="subhead">Extracted Images ({parseResult.images.length}) — auto-assigned</div>
                    <div className="grid grid-cols-2 gap-2">
                      {parseResult.images.map((img) => {
                        const assignOptions = [
                          ...(parseResult.npcs || []).map(n => ({ label: `NPC: ${n.name}`, value: n.name })),
                          ...(parseResult.scenes || []).map(s => ({ label: `Scene: ${s.title}`, value: s.title })),
                        ];
                        return (
                          <div key={img.idx} style={{ position: "relative" }}>
                            <img
                              src={img.url}
                              alt={img.label || `Image ${img.idx}`}
                              style={{ maxHeight: "120px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(img.url)}
                            />
                            <button
                              type="button"
                              onClick={() => deleteAssignedImage(img.idx)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                {img.assigned_to
                                  ? <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {img.assigned_to}</span>
                                  : <span style={{ color:"#7a6040" }}>{img.type || "illustration"}{img.label ? ` · ${img.label}` : ""}</span>
                                }
                              </div>
                              <select
                                value={img.assigned_to || ""}
                                onChange={e => assignOptions.length && assignImageTo(img.idx, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={assignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: assignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: assignOptions.length ? "pointer" : "default", opacity: assignOptions.length ? 1 : 0.5 }}
                              >
                                {assignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{assignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : images.embedded.length > 0 ? (
                  <>
                    <div className="subhead">Maps &amp; Artwork ({images.embedded.length})</div>
                    <div className="grid grid-cols-2 gap-2">
                      {images.embedded.map((url, i) => {
                        const assignedLabel = getAssignedLabel(url);
                        const npcIdx = (parseResult?.npcs || []).findIndex(n => n.image_url === url);
                        const sceneIdx = (parseResult?.scenes || []).findIndex(s => s.image_url === url);
                        const currentVal = npcIdx >= 0 ? `npc:${npcIdx}` : sceneIdx >= 0 ? `scene:${sceneIdx}` : "";
                        const embedAssignOptions = [
                          ...(parseResult?.npcs || []).map((n, ni) => ({ label: `NPC: ${n.name}`, value: `npc:${ni}` })),
                          ...(parseResult?.scenes || []).map((s, si) => ({ label: `Scene: ${s.title}`, value: `scene:${si}` })),
                        ];
                        return (
                          <div key={i} style={{ position: "relative" }}>
                            <img src={url} alt={`Image ${i + 1}`}
                              style={{ maxHeight: "130px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(url)} />
                            <button
                              type="button"
                              onClick={() => deleteEmbeddedImage(i)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              {assignedLabel && (
                                <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                  <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {assignedLabel.replace(/^(NPC|Scene): /, "")}</span>
                                </div>
                              )}
                              <select
                                value={currentVal}
                                onChange={e => handleAssignImage(url, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={embedAssignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: embedAssignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: embedAssignOptions.length ? "pointer" : "default", opacity: embedAssignOptions.length ? 1 : 0.5 }}
                              >
                                {embedAssignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{embedAssignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div className="intake-empty">Run "AI Parse" to automatically extract and assign images, or use "Extract Images" for raw extraction.</div>
                )}
                {images.pages.length > 0 && (
                  <>
                    <div className="subhead mt-2">Page Thumbnails ({images.pages.length})</div>
                    <div className="grid grid-cols-2 gap-2">
                      {images.pages.map((url, i) => {
                        const assignedLabel = getAssignedLabel(url);
                        const npcIdx = (parseResult?.npcs || []).findIndex(n => n.image_url === url);
                        const sceneIdx = (parseResult?.scenes || []).findIndex(s => s.image_url === url);
                        const currentVal = npcIdx >= 0 ? `npc:${npcIdx}` : sceneIdx >= 0 ? `scene:${sceneIdx}` : "";
                        const pageAssignOptions = [
                          ...(parseResult?.npcs || []).map((n, ni) => ({ label: `NPC: ${n.name}`, value: `npc:${ni}` })),
                          ...(parseResult?.scenes || []).map((s, si) => ({ label: `Scene: ${s.title}`, value: `scene:${si}` })),
                        ];
                        return (
                          <div key={i} style={{ position: "relative" }}>
                            <img src={url} alt={`Page ${i + 1}`}
                              style={{ maxHeight: "130px", width: "100%", borderRadius: "4px", border: "1px solid #4f341f", objectFit: "cover", cursor: "pointer", display: "block" }}
                              onClick={() => setLightbox(url)} />
                            <button
                              type="button"
                              onClick={() => deletePageImage(i)}
                              title="Remove image"
                              style={{ position:"absolute", top:"5px", right:"5px", width:"20px", height:"20px", borderRadius:"50%", background:"#1a0f06", border:"1px solid #c8a050", color:"#c8a050", fontSize:"12px", lineHeight:1, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", zIndex:2, transition:"border-color 0.15s, box-shadow 0.15s, color 0.15s" }}
                              onMouseEnter={e => { e.currentTarget.style.borderColor="#c05050"; e.currentTarget.style.color="#ff6b6b"; e.currentTarget.style.boxShadow="0 0 6px #c0505088"; }}
                              onMouseLeave={e => { e.currentTarget.style.borderColor="#c8a050"; e.currentTarget.style.color="#c8a050"; e.currentTarget.style.boxShadow="none"; }}
                            >×</button>
                            <div style={{ position:"absolute", bottom:0, left:0, right:0, background:"rgba(0,0,0,0.72)", borderBottomLeftRadius:"4px", borderBottomRightRadius:"4px", padding:"3px 5px" }}>
                              {assignedLabel && (
                                <div style={{ fontSize:"10px", marginBottom:"2px", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
                                  <span style={{ color:"#d4af37", fontWeight:600 }}>Linked to: {assignedLabel.replace(/^(NPC|Scene): /, "")}</span>
                                </div>
                              )}
                              <select
                                value={currentVal}
                                onChange={e => handleAssignImage(url, e.target.value)}
                                onClick={e => e.stopPropagation()}
                                disabled={pageAssignOptions.length === 0}
                                style={{ width:"100%", background:"#1a0f06", border:"1px solid #c8a050", color: pageAssignOptions.length ? "#c8a050" : "#5a3e1b", fontSize:"10px", borderRadius:"3px", padding:"1px 2px", cursor: pageAssignOptions.length ? "pointer" : "default", opacity: pageAssignOptions.length ? 1 : 0.5 }}
                              >
                                {pageAssignOptions.length === 0
                                  ? <option value="">Run AI Parse to assign</option>
                                  : <><option value="">Assign to...</option>{pageAssignOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</>
                                }
                              </select>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            )}

            {activePanel === "outline" && (
              <>
                {isAiResult && scenes.length > 0 && (() => {
                  const actGroups = scenes.reduce((acc, scene) => {
                    const act = scene.act || "Scenes";
                    if (!acc[act]) acc[act] = [];
                    acc[act].push(scene);
                    return acc;
                  }, {});
                  return (
                    <div className="space-y-1">
                      {Object.entries(actGroups).map(([act, actScenes]) => {
                        const isOpen = expandedActs.has(act);
                        const toggle = () => setExpandedActs(prev => {
                          const next = new Set(prev);
                          isOpen ? next.delete(act) : next.add(act);
                          return next;
                        });
                        return (
                          <div key={act} className="border border-[#4f341f] rounded overflow-hidden">
                            {/* Chapter header — clickable to expand/collapse */}
                            <button
                              type="button"
                              className="w-full flex items-center justify-between px-3 py-2 bg-[#1a0f06] hover:bg-[#2a1a0a] text-left"
                              onClick={toggle}
                            >
                              <span className="font-heading text-[#d4af37] text-sm">{act}</span>
                              <span className="text-[#9b7440] text-xs flex items-center gap-2">
                                <span>{actScenes.length} scene{actScenes.length !== 1 ? "s" : ""}</span>
                                <span className="text-base leading-none">{isOpen ? "▲" : "▼"}</span>
                              </span>
                            </button>

                            {/* Scenes — only visible when chapter is open */}
                            {isOpen && (
                              <ul className="divide-y divide-[#2a1a0a]">
                            {actScenes.map((scene, i) => {
                              const thumbSrc = scene.image_url || getScenePlaceholder(scene.type);
                              return (
                                  <li key={i}
                                    className="flex gap-2 items-start cursor-pointer hover:bg-[#2a1a0a] px-3 py-2"
                                    title="Click to view scene details"
                                    onClick={() => setDetailItem({type:"scene", data:scene})}>
                                      <img src={thumbSrc} alt={scene.title}
                                        className="w-12 h-10 object-cover rounded border border-[#4f341f] flex-shrink-0 mt-0.5"
                                        onClick={e => { e.stopPropagation(); if (scene.image_url) setLightbox(scene.image_url); }} />
                                    <div className="flex-1 min-w-0">
                                      <div className="font-semibold text-[#c8a050] text-sm">{scene.title}</div>
                                      <div className="flex gap-1 flex-wrap mt-0.5">
                                        {scene.type && <span className="text-[#9b7440] text-xs">({scene.type})</span>}
                                        {scene.difficulty && scene.difficulty !== "none" && (
                                          <span className={`text-xs ${scene.difficulty === "deadly" ? "text-red-400" : scene.difficulty === "hard" ? "text-orange-400" : "text-[#7a5a30]"}`}>{scene.difficulty}</span>
                                        )}
                                        {scene.location && <span className="text-xs text-[#7a5a30]">· {scene.location}</span>}
                                      </div>
                                      {scene.read_aloud && (
                                        <div className="text-xs text-[#7a5a30] italic mt-0.5 line-clamp-2">{scene.read_aloud.slice(0, 100)}…</div>
                                      )}
                                    </div>
                                    <span className="text-[#4f341f] text-xs flex-shrink-0 mt-1">›</span>
                                  </li>
                                );})}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
                {!isAiResult && acts.length > 0 && (() => {
                  return (
                    <div className="space-y-1">
                      {acts.map((act) => {
                        const isOpen = expandedActs.has(act.title);
                        const toggle = () => setExpandedActs(prev => {
                          const next = new Set(prev);
                          isOpen ? next.delete(act.title) : next.add(act.title);
                          return next;
                        });
                        return (
                          <div key={act.title} className="border border-[#4f341f] rounded overflow-hidden">
                            <button type="button"
                              className="w-full flex items-center justify-between px-3 py-2 bg-[#1a0f06] hover:bg-[#2a1a0a] text-left"
                              onClick={toggle}>
                              <span className="font-heading text-[#d4af37] text-sm">{act.title}</span>
                              <span className="text-[#9b7440] text-xs">{isOpen ? "▲" : "▼"}</span>
                            </button>
                            {isOpen && (
                              <ul className="divide-y divide-[#2a1a0a]">
                                {(act.scenes || []).map((scene, i) => (
                                  <li key={i} className="px-3 py-1.5 text-sm text-[#c8a050]">{scene}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
              </>
            )}

            {activePanel === "items" && (
              <div className="space-y-2">
                {items.length ? items.map((item, i) => (
                  <div key={item.name || i} className="intake-act-card cursor-pointer hover:border-[#d4af37]"
                    onClick={() => setDetailItem({type:"item", data:item})}>
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="truncate">{item.name}</h3>
                      {item.magical && <span className="text-xs border border-amber-800 rounded px-1 text-amber-400 flex-shrink-0">magical</span>}
                    </div>
                    {item.scene && <p className="text-xs text-[#7a5a30]">{item.scene}</p>}
                    {item.description && <p className="text-xs text-[#9b7440] mt-0.5">{item.description}</p>}
                  </div>
                )) : (
                  <div className="intake-empty">No items/treasure extracted yet. Run AI Parse on an adventure with loot.</div>
                )}
              </div>
            )}

            {activePanel === "locations" && (
              <div className="space-y-2">
                {locations.length ? locations.filter(Boolean).map((loc) => (
                  <div key={loc.name || String(loc)} className="intake-act-card cursor-pointer hover:border-[#d4af37]"
                    onClick={() => isAiResult && setDetailItem({type:"location", data:loc})}>
                    <h3>{loc.name || loc}</h3>
                    {loc.description && <p className="text-xs text-[#9b7440]">{loc.description}</p>}
                  </div>
                )) : (
                  <div className="intake-empty">No locations extracted yet.</div>
                )}
                {!isAiResult && parseResult?.locations?.length > 0 && (
                  <ul className="intake-pill-list">
                    {parseResult.locations.map(l => <li key={l}>{l}</li>)}
                  </ul>
                )}
              </div>
            )}

            {activePanel === "party" && (
              <div className="space-y-2">
                {party.length ? party.map((pc) => (
                  <div key={pc.name} className="intake-act-card">
                    <h3>{pc.name}</h3>
                    <p className="text-xs text-[#9b7440]">
                      {[pc.race, pc.class_].filter(Boolean).join(" ")}
                      {pc.level ? ` · Lv ${pc.level}` : ""}
                      {pc.hp ? ` · HP ${pc.hp}` : ""}
                      {pc.ac ? ` · AC ${pc.ac}` : ""}
                    </p>
                  </div>
                )) : (
                  <div className="intake-empty">No player characters found in the uploaded docs.</div>
                )}
              </div>
            )}

            {activePanel === "review" && (
              <ExtractionReviewQueue
                documentName={parseResult?.title || (files.length === 1 ? files[0].name : undefined)}
              />
            )}
          </PrepPanel>
        </div>

        {/* Right: NPCs + Reveals */}
        <div className="xl:col-span-5 min-h-0">
          <PrepPanel title="Extracted Campaign Data" className="h-full">

            <div className="subhead">NPC Roster</div>
            {isAiResult && npcs.length ? (
              <div className="space-y-2 mb-4">
                {npcs.slice(0, 8).map((npc) => (
                  <div key={npc.name} className="intake-act-card flex gap-2 cursor-pointer hover:border-[#d4af37]"
                    onClick={() => setDetailItem({type:"npc", data:npc})}>
                    {npc.image_url ? (
                      <img
                        src={npc.image_url}
                        alt={npc.name}
                        className="w-14 h-14 object-cover rounded border border-[#4f341f] flex-shrink-0"
                        onClick={e => { e.stopPropagation(); setLightbox(npc.image_url); }}
                      />
                    ) : (
                      <div className="w-14 h-14 flex-shrink-0 rounded border border-[#4f341f] bg-[#1a0f06] flex items-center justify-center text-[#4f341f] text-lg font-bold">
                        {(npc.name || "?").slice(0, 2).toUpperCase()}
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <h3 className="truncate">{npc.name}</h3>
                        <span className="text-xs text-[#9b7440] border border-[#4f341f] rounded px-1 flex-shrink-0">{typeof npc.role === "string" ? npc.role : ""}</span>
                      </div>
                      {npc.faction && <p className="text-xs text-[#7a5a30]">{typeof npc.faction === "string" ? npc.faction : ""}</p>}
                      {npc.personality && typeof npc.personality === "string" && <p className="text-xs text-[#9b7440] italic mt-0.5">{npc.personality.slice(0, 80)}{npc.personality.length > 80 ? "…" : ""}</p>}
                      {npc.motivation && typeof npc.motivation === "string" && <p className="text-xs text-[#7a5a30]">Wants: {npc.motivation.slice(0, 60)}{npc.motivation.length > 60 ? "…" : ""}</p>}
                    </div>
                  </div>
                ))}
              </div>
            ) : !isAiResult && parseResult?.npcs?.length ? (
              <ul className="intake-pill-list mb-4">
                {parseResult.npcs.map(n => <li key={n}>{n}</li>)}
              </ul>
            ) : (
              <div className="intake-empty mb-3">No NPCs extracted yet.</div>
            )}

            <div className="subhead">Reveals &amp; Plot Hooks</div>
            {reveals.length ? (
              <div className="prep-reveal-list">
                {reveals.map((reveal) => (
                  <div key={reveal.name || reveal} className="prep-reveal-row cursor-pointer hover:bg-[#2a1a0a] rounded px-1 -mx-1"
                    onClick={() => isAiResult && setDetailItem({type:"reveal", data:reveal})}>
                    <span className={`prep-reveal-dot ${TYPE_BADGE[reveal.type] || "amber"}`} />
                    <span className="prep-reveal-name">{reveal.name || reveal}</span>
                    <span className="prep-reveal-status text-xs">{reveal.when || reveal.type || ""}</span>
                  </div>
                ))}
              </div>
            ) : !isAiResult && parseResult?.reveals?.length ? (
              <ul className="intake-pill-list">
                {parseResult.reveals.map(r => <li key={r}>{r}</li>)}
              </ul>
            ) : (
              <div className="intake-empty">No reveals extracted yet.</div>
            )}
          </PrepPanel>
        </div>

      </section>

      {/* Detail drawer */}
      <DetailDrawer item={detailItem} onClose={() => setDetailItem(null)} onLightbox={setLightbox} />

      {/* Lightbox overlay */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 cursor-pointer"
          onClick={() => setLightbox(null)}
        >
          <img
            src={lightbox}
            alt="Full size"
            className="max-w-[90vw] max-h-[90vh] rounded border-2 border-[#d4af37] shadow-2xl"
            onClick={e => e.stopPropagation()}
          />
          <button
            type="button"
            className="absolute top-4 right-6 text-[#d4af37] text-3xl font-bold"
            onClick={() => setLightbox(null)}
          >
            ×
          </button>
        </div>
      )}
    </div>
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
          <PrepRoom view={view} onNavigate={onNavigate} campaignData={campaignData} onUpdateCampaign={setCampaignData} />
        }
        libraryContent={
          <ErrorBoundary>
            <AdventureIntake
              view={view}
              onNavigate={onNavigate}
              campaignData={campaignData}
              onSaveCampaign={setCampaignData}
              authFetch={authFetch}
            />
          </ErrorBoundary>
        }
        onNavigate={onNavigate}
      />
    );
  } else if (view === "intake") {
    // Backward-compat: direct /intake URL still works
    content = (
      <ErrorBoundary>
        <AdventureIntake
          view={view}
          onNavigate={onNavigate}
          campaignData={campaignData}
          onSaveCampaign={setCampaignData}
          authFetch={authFetch}
        />
      </ErrorBoundary>
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
    content = <CampaignPage />;
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
