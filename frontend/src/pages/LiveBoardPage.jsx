import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ChevronLeft, ChevronRight, Play, SkipForward, Trash2 } from "lucide-react";
import WorkspaceContainer from "../components/layout/WorkspaceContainer";
import NpcVoiceModal from "../components/live-board/NpcVoiceModal";
import SessionLog from "../components/live-board/SessionLog";
import { FantasyButton } from "../components/shared";
import { getVoices } from "../lib/api/voices";
import { createId } from "../lib/utils/ids";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  useActiveCampaign,
  useScenesForActiveCampaign,
  useSceneNpcs,
  useSceneCodexEntries,
  useCodexEntriesForActiveCampaign,
  useVoices,
  useActionLogForActiveScene,
} from "../store/selectors";

function readAloudText(scene) {
  if (!scene) return "";
  const v = scene.readAloud ?? scene.read_aloud ?? scene.summary ?? scene.notes;
  return String(v || "").trim();
}

/** Map store ActionLogEvent → SessionLogEntry shape (role/text/meta). */
function mapLogForSessionUi(entries) {
  const roleForType = {
    player: "player",
    npc: "lore",
    narration: "lore",
    system: "error",
    gm_note: "lore",
  };
  return entries.map((e) => ({
    id: e.id,
    role: roleForType[e.type] ?? "lore",
    text: e.text,
    meta: e.createdAt ? String(e.createdAt).slice(11, 19) : "",
  }));
}

/**
 * Live Board — one active scene from Zustand, TTS, NPC speak modal, initiative, scene-linked codex / encounters, session log.
 */
export default function LiveBoardPage() {
  const activeCampaign = useActiveCampaign();
  const scenes = useScenesForActiveCampaign();
  const activeSceneId = useCampaignContextStore((s) => s.activeSceneId);
  const setActiveScene = useCampaignContextStore((s) => s.setActiveScene);
  const upsertVoice = useCampaignContextStore((s) => s.upsertVoice);
  const addActionLogEvent = useCampaignContextStore((s) => s.addActionLogEvent);

  const voices = useVoices();
  const actionLog = useActionLogForActiveScene();
  const allCodex = useCodexEntriesForActiveCampaign();

  const [requireApiKey, setRequireApiKey] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const [narrateBusy, setNarrateBusy] = useState(false);
  const [narrateError, setNarrateError] = useState("");

  const [speakNpc, setSpeakNpc] = useState(null);
  const [speakLine, setSpeakLine] = useState("");
  const [speakBusy, setSpeakBusy] = useState(false);
  const [speakError, setSpeakError] = useState("");

  /** @type {[{ id: string, name: string, initiative: number }]} */
  const [initiativeRows, setInitiativeRows] = useState([]);
  const [turnIndex, setTurnIndex] = useState(0);
  const [newInitName, setNewInitName] = useState("");
  const [newInitValue, setNewInitValue] = useState("");

  const authFetch = useCallback(
    (input, init = {}) => {
      const headers = new Headers(init.headers || {});
      const key = apiKey.trim();
      if (key) headers.set("X-API-Key", key);
      return fetch(input, { ...init, headers });
    },
    [apiKey]
  );

  useEffect(() => {
    let cancelled = false;
    fetch("/config")
      .then((r) => (r.ok ? r.json() : { require_api_key: false }))
      .then((cfg) => {
        if (!cancelled) setRequireApiKey(Boolean(cfg?.require_api_key));
      })
      .catch(() => {
        if (!cancelled) setRequireApiKey(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await getVoices(authFetch);
      if (cancelled || !list.length) return;
      const cid = useCampaignContextStore.getState().activeCampaignId;
      for (const p of list) {
        const id = String(p.voice_id || p.id || "");
        if (!id) continue;
        upsertVoice({
          id,
          campaignId: cid || undefined,
          name: p.name || "Voice",
          tags: Array.isArray(p.tags) ? p.tags : [],
          assignedNpcIds: [],
          tone: p.tone,
          accent: p.accent,
          status: p.status || "ready",
        });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [authFetch, upsertVoice, activeCampaign?.id]);

  useEffect(() => {
    if (!scenes.length) return;
    const valid = activeSceneId && scenes.some((s) => s.id === activeSceneId);
    if (!valid) setActiveScene(scenes[0].id);
  }, [scenes, activeSceneId, setActiveScene]);

  const activeScene = useMemo(
    () => scenes.find((s) => s.id === activeSceneId) ?? null,
    [scenes, activeSceneId]
  );

  const sceneNpcs = useSceneNpcs(activeScene?.id);
  const sceneCodex = useSceneCodexEntries(activeScene?.id);

  const statBlocks = useMemo(() => {
    if (!activeScene) return [];
    const titleKey = (activeScene.title || activeScene.name || "").trim().toLowerCase();
    const byId = new Map();
    for (const e of sceneCodex) byId.set(e.id, e);
    for (const e of allCodex) {
      const tags = (e.tags || []).map((t) => String(t).toLowerCase());
      if (!tags.some((t) => t.includes("encounter"))) continue;
      const linked =
        e.relatedSceneIds?.includes(activeScene.id) ||
        (titleKey &&
          (e.relatedSceneNames || []).some((n) => String(n).trim().toLowerCase() === titleKey));
      if (linked) byId.set(e.id, e);
    }
    return [...byId.values()];
  }, [sceneCodex, allCodex, activeScene]);

  const sceneIndex = useMemo(
    () => (activeScene ? scenes.findIndex((s) => s.id === activeScene.id) : -1),
    [scenes, activeScene]
  );

  const narrateVoiceId = useMemo(() => {
    if (!activeScene) return "";
    const nv =
      activeScene.narratorVoiceId ||
      activeScene.narrator_voice_id ||
      (voices[0] && voices[0].id) ||
      "";
    return String(nv || "");
  }, [activeScene, voices]);

  const sortedInitiative = useMemo(
    () => [...initiativeRows].sort((a, b) => b.initiative - a.initiative),
    [initiativeRows]
  );

  useEffect(() => {
    setInitiativeRows([]);
    setTurnIndex(0);
    setNewInitName("");
    setNewInitValue("");
  }, [activeScene?.id]);

  useEffect(() => {
    const n = sortedInitiative.length;
    setTurnIndex((i) => (n === 0 ? 0 : Math.min(Math.max(0, i), n - 1)));
  }, [sortedInitiative.length]);

  const playAudioBlob = useCallback(async (blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    await new Promise((resolve, reject) => {
      audio.onended = () => {
        URL.revokeObjectURL(url);
        resolve();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Audio playback failed"));
      };
      audio.play().catch(reject);
    });
  }, []);

  const handleNarrateScene = useCallback(async () => {
    const text = readAloudText(activeScene);
    if (!text || !narrateVoiceId) {
      setNarrateError(!text ? "No read-aloud text for this scene." : "Assign a narrator voice in Prep or add voices.");
      return;
    }
    setNarrateBusy(true);
    setNarrateError("");
    try {
      const res = await authFetch("/tts/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice_id: narrateVoiceId }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || err.error || res.statusText || "TTS failed");
      }
      const blob = await res.blob();
      await playAudioBlob(blob);
      addActionLogEvent({
        type: "narration",
        text: `Read-aloud played: ${(activeScene.title || "Scene").slice(0, 80)}`,
      });
    } catch (e) {
      setNarrateError(e?.message || "TTS failed");
    } finally {
      setNarrateBusy(false);
    }
  }, [activeScene, narrateVoiceId, authFetch, playAudioBlob, addActionLogEvent]);

  const handleSpeakSubmit = useCallback(async () => {
    const line = (speakLine || "").trim();
    if (!speakNpc || !line) return;
    setSpeakBusy(true);
    setSpeakError("");
    try {
      const res = await authFetch("/tts/npc-dialogue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ npc_id: speakNpc.id, text: line }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || err.error || res.statusText || "NPC dialogue failed");
      }
      const blob = await res.blob();
      await playAudioBlob(blob);
      addActionLogEvent({
        type: "npc",
        text: `${speakNpc.name}: ${line.slice(0, 200)}`,
      });
      setSpeakNpc(null);
      setSpeakLine("");
    } catch (e) {
      setSpeakError(e?.message || "NPC dialogue failed");
    } finally {
      setSpeakBusy(false);
    }
  }, [speakNpc, speakLine, authFetch, playAudioBlob, addActionLogEvent]);

  const goPrevScene = () => {
    if (sceneIndex <= 0) return;
    setActiveScene(scenes[sceneIndex - 1].id);
  };

  const goNextScene = () => {
    if (sceneIndex < 0 || sceneIndex >= scenes.length - 1) return;
    setActiveScene(scenes[sceneIndex + 1].id);
  };

  const addInitiativeRow = () => {
    const name = newInitName.trim();
    const init = parseInt(String(newInitValue).trim(), 10);
    if (!name || !Number.isFinite(init)) return;
    setInitiativeRows((rows) => [...rows, { id: createId("init"), name, initiative: init }]);
    setNewInitName("");
    setNewInitValue("");
  };

  const removeInitiativeRow = (id) => {
    setInitiativeRows((rows) => rows.filter((r) => r.id !== id));
  };

  const nextTurn = () => {
    if (sortedInitiative.length === 0) return;
    setTurnIndex((i) => (i + 1) % sortedInitiative.length);
  };

  const logUi = useMemo(() => mapLogForSessionUi(actionLog), [actionLog]);

  if (!activeCampaign) {
    return (
      <WorkspaceContainer className="live-board dm-ghost-shell">
        <div className="min-h-0 flex-1 flex flex-col gap-4 pb-4">
          <section className="panel-ornate rounded-lg overflow-hidden flex-shrink-0 border-[#2a1a08] bg-[#120c08]/90">
            <div className="panel-head panel-head--row flex-wrap gap-2 border-[#2a1a08]">
              <div className="plaque text-[#e7c27a]/80">Active scene</div>
              <span className="text-[10px] uppercase tracking-[0.12em] text-[#5c4a38]">Waiting for campaign</span>
            </div>
            <div className="panel-body space-y-2 border-[#2a1a08]">
              <div className="dm-ghost-border min-h-[100px] px-3 py-3">
                <p className="dm-ghost-hint text-xs leading-relaxed">
                  Scene title and read-aloud will show here once you load a campaign and select a scene.
                </p>
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0 flex-1">
            <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[200px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">Party roster</div>
              </div>
              <div className="panel-body flex-1 flex flex-col gap-2 border-[#2a1a08]">
                <div className="dm-ghost-border dm-ghost-row px-2 shrink-0" />
                <div className="dm-ghost-border dm-ghost-row px-2 shrink-0" />
                <p className="dm-ghost-hint text-xs mt-auto pt-1">
                  Party members will appear here when you track them in session.
                </p>
              </div>
            </section>

            <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[200px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">NPCs in scene</div>
              </div>
              <div className="panel-body flex-1 flex flex-col gap-2 border-[#2a1a08]">
                <div className="dm-ghost-border min-h-[88px] px-3 py-2 flex items-center">
                  <p className="dm-ghost-hint text-xs">NPCs will appear when a scene is active.</p>
                </div>
              </div>
            </section>

            <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[200px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head panel-head--row border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">Initiative</div>
              </div>
              <div className="panel-body flex-1 flex flex-col gap-2 border-[#2a1a08]">
                <div className="dm-ghost-border min-h-[100px] px-3 py-3">
                  <p className="dm-ghost-hint text-xs mb-2">Add combatants to start initiative.</p>
                  <div className="dm-ghost-border border-[#2a2014] bg-[rgba(14,10,6,0.5)] h-8 rounded opacity-70" />
                </div>
              </div>
            </section>
          </div>

          <section className="panel-ornate rounded-lg overflow-hidden flex flex-col max-h-[280px] border-[#2a1a08] bg-[#120c08]/90">
            <div className="panel-head border-[#2a1a08]">
              <div className="plaque text-[#e7c27a]/80">Encounters &amp; stat blocks</div>
            </div>
            <div className="panel-body overflow-y-auto space-y-2 border-[#2a1a08]">
              <div className="dm-ghost-border min-h-[72px] px-3 py-2">
                <p className="dm-ghost-hint text-xs">
                  Linked codex encounters and stat blocks will populate here for the active scene.
                </p>
              </div>
            </div>
          </section>

          <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[140px] max-h-[220px] border-[#2a1a08] bg-[#120c08]/90">
            <div className="panel-head flex-shrink-0 border-[#2a1a08]">
              <div className="plaque text-[#e7c27a]/80">Session log</div>
            </div>
            <div className="panel-body flex-1 border-[#2a1a08]">
              <div className="dm-ghost-border min-h-[72px] px-3 py-2 h-full">
                <p className="dm-ghost-hint text-xs">Narration and dialogue log will stream here during play.</p>
              </div>
            </div>
          </section>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2 pb-4">
            <Link
              to="/library"
              className="text-sm font-heading text-[#e7c27a] border border-dashed border-[#5c4a38] rounded px-4 py-2 hover:border-[#9b7440] hover:bg-[#1a1208]/80 transition-colors"
            >
              Load a campaign from Library →
            </Link>
          </div>
        </div>
      </WorkspaceContainer>
    );
  }

  if (!scenes.length || !activeScene) {
    return (
      <WorkspaceContainer className="live-board dm-ghost-shell">
        <div className="min-h-0 flex-1 flex flex-col gap-4 pb-4">
          <section className="panel-ornate rounded-lg overflow-hidden flex-shrink-0 border-[#2a1a08] bg-[#120c08]/90">
            <div className="panel-head panel-head--row border-[#2a1a08]">
              <div className="plaque text-[#e7c27a]/80">Active scene</div>
            </div>
            <div className="panel-body border-[#2a1a08]">
              <div className="dm-ghost-border min-h-[80px] px-3 py-3">
                <p className="dm-ghost-hint text-xs">
                  No scenes in this campaign yet. Add scenes in Prep or build them from the Library.
                </p>
              </div>
            </div>
          </section>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1">
            <section className="panel-ornate rounded-lg overflow-hidden min-h-[160px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">Party roster</div>
              </div>
              <div className="panel-body border-[#2a1a08]">
                <div className="dm-ghost-border min-h-[64px] px-2 py-2">
                  <p className="dm-ghost-hint text-xs">Party slots will appear here.</p>
                </div>
              </div>
            </section>
            <section className="panel-ornate rounded-lg overflow-hidden min-h-[160px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">NPCs in scene</div>
              </div>
              <div className="panel-body border-[#2a1a08]">
                <div className="dm-ghost-border min-h-[64px] px-2 py-2">
                  <p className="dm-ghost-hint text-xs">NPCs will appear when a scene is active.</p>
                </div>
              </div>
            </section>
            <section className="panel-ornate rounded-lg overflow-hidden min-h-[160px] border-[#2a1a08] bg-[#120c08]/90">
              <div className="panel-head border-[#2a1a08]">
                <div className="plaque text-[#e7c27a]/80">Initiative</div>
              </div>
              <div className="panel-body border-[#2a1a08]">
                <div className="dm-ghost-border min-h-[64px] px-2 py-2">
                  <p className="dm-ghost-hint text-xs">Add combatants to start initiative.</p>
                </div>
              </div>
            </section>
          </div>
          <section className="panel-ornate rounded-lg overflow-hidden border-[#2a1a08] bg-[#120c08]/90">
            <div className="panel-head border-[#2a1a08]">
              <div className="plaque text-[#e7c27a]/80">Encounters &amp; stat blocks</div>
            </div>
            <div className="panel-body border-[#2a1a08]">
              <div className="dm-ghost-border min-h-[56px] px-2 py-2">
                <p className="dm-ghost-hint text-xs">Encounters show here when linked to a scene.</p>
              </div>
            </div>
          </section>
          <div className="flex flex-wrap justify-center gap-3">
            <Link to="/prep" className="text-sm font-heading text-[#e7c27a] underline underline-offset-2">
              Open Prep →
            </Link>
            <Link to="/library" className="text-sm font-heading text-[#9b7440] hover:text-[#e7c27a] underline underline-offset-2">
              Library →
            </Link>
          </div>
        </div>
      </WorkspaceContainer>
    );
  }

  const readOut = readAloudText(activeScene);

  return (
    <WorkspaceContainer className="live-board">
      <div className="min-h-0 flex-1 flex flex-col gap-4 pb-4">
        {requireApiKey && (
          <div className="rounded border border-[#4a3018] bg-[#110b06] px-3 py-2">
            <label className="text-[10px] text-[var(--text-2)] uppercase tracking-wide block mb-1">
              API key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full max-w-md rounded border border-[#4a3018] bg-[#1c1008] px-2 py-1 text-xs text-[var(--text-1)]"
              autoComplete="off"
            />
          </div>
        )}

        {/* Scene navigation + header */}
        <section className="panel-ornate rounded-lg overflow-hidden flex-shrink-0">
          <div className="panel-head panel-head--row flex-wrap gap-2">
            <div className="flex items-center gap-1">
              <button
                type="button"
                className="p-1.5 rounded border border-[#5c3e23] text-[var(--gold)] disabled:opacity-30"
                onClick={goPrevScene}
                disabled={sceneIndex <= 0}
                aria-label="Previous scene"
              >
                <ChevronLeft size={18} />
              </button>
              <button
                type="button"
                className="p-1.5 rounded border border-[#5c3e23] text-[var(--gold)] disabled:opacity-30"
                onClick={goNextScene}
                disabled={sceneIndex < 0 || sceneIndex >= scenes.length - 1}
                aria-label="Next scene"
              >
                <ChevronRight size={18} />
              </button>
            </div>
            <div className="plaque flex-1 min-w-0">
              <span className="truncate">{activeScene.title || activeScene.name || "Scene"}</span>
              <span className="block text-[10px] uppercase tracking-[0.12em] text-[#9b7440] mt-0.5">
                Scene {sceneIndex + 1} / {scenes.length}
              </span>
            </div>
            {activeScene.location ? (
              <span className="text-[10px] uppercase tracking-[0.16em] text-[#9b7440]">{activeScene.location}</span>
            ) : null}
          </div>
          <div className="panel-body space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1">Read-aloud</div>
                {readOut ? (
                  <div className="read-aloud-text rounded-sm text-sm leading-relaxed">{readOut}</div>
                ) : (
                  <p className="text-xs text-[var(--text-2)] italic">No read-aloud text.</p>
                )}
              </div>
              <FantasyButton
                variant="secondary"
                className="text-xs shrink-0 inline-flex items-center gap-1.5"
                onClick={handleNarrateScene}
                disabled={narrateBusy || !readOut || !narrateVoiceId}
              >
                <Play size={14} />
                {narrateBusy ? "Playing…" : "Play TTS"}
              </FantasyButton>
            </div>
            {narrateError ? <div className="text-xs text-red-400">{narrateError}</div> : null}
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0 flex-1">
          {/* NPC roster */}
          <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[200px]">
            <div className="panel-head">
              <div className="plaque">NPC roster</div>
            </div>
            <div className="panel-body flex-1 overflow-y-auto space-y-2">
              {sceneNpcs.length === 0 ? (
                <p className="text-xs text-[var(--text-2)]">No NPCs linked to this scene.</p>
              ) : (
                sceneNpcs.map((npc) => (
                  <button
                    key={npc.id}
                    type="button"
                    onClick={() => {
                      setSpeakNpc(npc);
                      setSpeakLine("");
                      setSpeakError("");
                    }}
                    className="w-full text-left rounded border border-[#5c3e23] bg-[#1a1008]/80 px-3 py-2 hover:border-[var(--gold)]/50 transition-colors"
                  >
                    <div className="font-heading text-sm text-[var(--text-1)]">{npc.name}</div>
                    {npc.role ? <div className="text-[10px] text-[var(--text-2)]">{npc.role}</div> : null}
                    {npc.summary ? (
                      <div className="text-[11px] text-[#9b7440] mt-1 line-clamp-2">{npc.summary}</div>
                    ) : null}
                  </button>
                ))
              )}
            </div>
          </section>

          {/* Initiative */}
          <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[200px]">
            <div className="panel-head panel-head--row">
              <div className="plaque">Initiative</div>
              <FantasyButton
                variant="ghost"
                className="text-[10px] px-2 py-0.5 inline-flex items-center gap-1"
                onClick={nextTurn}
                disabled={sortedInitiative.length === 0}
              >
                <SkipForward size={12} />
                Next turn
              </FantasyButton>
            </div>
            <div className="panel-body flex-1 flex flex-col gap-2 min-h-0">
              <div className="flex flex-wrap gap-2 items-end">
                <label className="flex flex-col gap-0.5 text-[10px] text-[var(--text-2)]">
                  Name
                  <input
                    className="rounded border border-[#4a3018] bg-[#1c1008] px-2 py-1 text-xs text-[var(--text-1)] w-[140px]"
                    value={newInitName}
                    onChange={(e) => setNewInitName(e.target.value)}
                    placeholder="Creature / PC"
                  />
                </label>
                <label className="flex flex-col gap-0.5 text-[10px] text-[var(--text-2)]">
                  Init
                  <input
                    type="number"
                    className="rounded border border-[#4a3018] bg-[#1c1008] px-2 py-1 text-xs text-[var(--text-1)] w-20"
                    value={newInitValue}
                    onChange={(e) => setNewInitValue(e.target.value)}
                    placeholder="18"
                  />
                </label>
                <FantasyButton variant="secondary" className="text-xs py-1" onClick={addInitiativeRow}>
                  Add
                </FantasyButton>
              </div>
              <ul className="flex-1 overflow-y-auto space-y-1 text-xs">
                {sortedInitiative.length === 0 ? (
                  <li className="text-[var(--text-2)] italic">Add combatants to track turns.</li>
                ) : (
                  sortedInitiative.map((row, i) => (
                    <li
                      key={row.id}
                      className={[
                        "flex items-center justify-between gap-2 rounded border px-2 py-1.5",
                        i === turnIndex
                          ? "border-[var(--gold)] bg-[rgba(202,167,75,0.12)]"
                          : "border-[#3a2510] bg-[#130c06]",
                      ].join(" ")}
                    >
                      <span className="tabular-nums text-[var(--gold)] w-8">{row.initiative}</span>
                      <span className="flex-1 truncate text-[var(--text-1)]">{row.name}</span>
                      <button
                        type="button"
                        className="p-1 text-red-400/80 hover:text-red-300"
                        onClick={() => removeInitiativeRow(row.id)}
                        aria-label={`Remove ${row.name}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    </li>
                  ))
                )}
              </ul>
            </div>
          </section>
        </div>

        {/* Encounters / stat blocks */}
        <section className="panel-ornate rounded-lg overflow-hidden flex flex-col max-h-[320px]">
          <div className="panel-head">
            <div className="plaque">Encounters &amp; stat blocks</div>
          </div>
          <div className="panel-body overflow-y-auto space-y-3">
            {statBlocks.length === 0 ? (
              <p className="text-xs text-[var(--text-2)]">
                No codex entries linked to this scene. Link entries in Prep or import encounters tagged for this scene.
              </p>
            ) : (
              statBlocks.map((entry) => (
                <article
                  key={entry.id}
                  className="rounded border border-[#c79f5b]/40 bg-[#0e1a0e]/40 p-3"
                >
                  <h3 className="font-heading text-sm text-[#ffe08a]">{entry.title}</h3>
                  {entry.summary ? (
                    <p className="text-xs text-[#c8a97a] mt-1 whitespace-pre-wrap">{entry.summary}</p>
                  ) : null}
                  {entry.content ? (
                    <pre className="mt-2 text-[11px] text-[#d4f0cf] whitespace-pre-wrap font-mono leading-relaxed">
                      {entry.content}
                    </pre>
                  ) : null}
                </article>
              ))
            )}
          </div>
        </section>

        {/* Session log */}
        <section className="panel-ornate rounded-lg overflow-hidden flex flex-col min-h-[160px] max-h-[260px]">
          <div className="panel-head flex-shrink-0">
            <div className="plaque">Session log</div>
          </div>
          <div className="flex-1 min-h-0 flex flex-col px-0 pb-0">
            <SessionLog actionLog={logUi} />
          </div>
        </section>
      </div>

      <NpcVoiceModal
        open={Boolean(speakNpc)}
        mode="speak"
        npc={speakNpc}
        value={speakLine}
        onChange={setSpeakLine}
        onClose={() => {
          if (!speakBusy) {
            setSpeakNpc(null);
            setSpeakLine("");
            setSpeakError("");
          }
        }}
        onSubmit={handleSpeakSubmit}
        busy={speakBusy}
        error={speakError}
      />
    </WorkspaceContainer>
  );
}
