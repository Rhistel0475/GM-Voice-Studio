import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Mic2 } from "lucide-react";
import PrepVoiceCloneModal from "../components/prep/PrepVoiceCloneModal";
import { getVoices } from "../lib/api/voices";
import { persistNpcVoice, persistSceneContent } from "../lib/campaignPersistence";
import { useCampaignContextStore } from "../store/campaignContext";
import {
  useActiveCampaign,
  useScenesForActiveCampaign,
  useSceneNpcs,
  useVoices,
} from "../store/selectors";

function getReadAloud(scene) {
  if (!scene) return "";
  const v = scene.readAloud ?? scene.read_aloud;
  return typeof v === "string" ? v : "";
}

function getGmNotes(scene) {
  if (!scene) return "";
  return typeof scene.notes === "string" ? scene.notes : "";
}

/**
 * Prep room: scenes from Zustand campaign context (selectors only).
 * Left = scene list; right = read-aloud, NPC voices, GM notes. Saves → store + PATCH APIs.
 */
export default function PrepPage() {
  const activeCampaign = useActiveCampaign();
  const scenes = useScenesForActiveCampaign();
  const voices = useVoices();

  const activeSceneId = useCampaignContextStore((s) => s.activeSceneId);
  const setActiveScene = useCampaignContextStore((s) => s.setActiveScene);
  const upsertScene = useCampaignContextStore((s) => s.upsertScene);
  const assignVoiceToNpc = useCampaignContextStore((s) => s.assignVoiceToNpc);
  const unassignVoiceFromNpc = useCampaignContextStore((s) => s.unassignVoiceFromNpc);
  const upsertVoice = useCampaignContextStore((s) => s.upsertVoice);

  const [requireApiKey, setRequireApiKey] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [cloneOpen, setCloneOpen] = useState(false);
  const [readDraft, setReadDraft] = useState("");
  const [notesDraft, setNotesDraft] = useState("");

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

  /** Hydrate voice library from API into the store for dropdowns. */
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

  const selectedScene = useMemo(
    () => scenes.find((s) => s.id === activeSceneId) ?? null,
    [scenes, activeSceneId]
  );

  const sceneNpcs = useSceneNpcs(selectedScene?.id);

  useEffect(() => {
    if (!selectedScene) {
      setReadDraft("");
      setNotesDraft("");
      return;
    }
    setReadDraft(getReadAloud(selectedScene));
    setNotesDraft(getGmNotes(selectedScene));
  }, [selectedScene?.id]);

  const flushSceneContent = useCallback(() => {
    if (!selectedScene) return;
    const title = (selectedScene.title || selectedScene.name || "").trim();
    if (!title) return;
    const read = readDraft;
    const notes = notesDraft;
    upsertScene({
      ...selectedScene,
      readAloud: read,
      read_aloud: read,
      notes,
    });
    persistSceneContent(authFetch, title, { readAloud: read, notes });
  }, [selectedScene, readDraft, notesDraft, upsertScene, authFetch]);

  const handleVoiceChange = useCallback(
    (npc, voiceId) => {
      if (voiceId) assignVoiceToNpc(npc.id, voiceId);
      else unassignVoiceFromNpc(npc.id);
      persistNpcVoice(authFetch, npc.name, voiceId || "");
    },
    [assignVoiceToNpc, unassignVoiceFromNpc, authFetch]
  );

  if (!activeCampaign) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4 px-6 text-center">
        <p className="text-sm text-neutral-600">No active campaign in the workspace.</p>
        <Link
          to="/import"
          className="text-sm font-medium text-amber-800 hover:text-amber-950 underline"
        >
          Import or apply a campaign →
        </Link>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 overflow-hidden gap-0 bg-[#0d0804]">
      <aside
        className="flex-shrink-0 flex flex-col border-r border-[#3a2510] overflow-y-auto bg-[#0d0804]"
        style={{ width: "260px" }}
      >
        <div className="p-3 border-b border-[#2a1a0a]">
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] font-heading">
            Scenes
          </div>
          <div className="font-heading text-sm text-[var(--gold)] mt-1 truncate" title={activeCampaign.name}>
            {activeCampaign.name}
          </div>
        </div>
        <div className="flex-1 p-2 space-y-0.5">
          {scenes.length === 0 ? (
            <p className="text-xs text-[var(--text-2)] px-2 py-4 text-center leading-relaxed">
              No scenes yet. Import an adventure or add scenes from your workflow.
            </p>
          ) : (
            scenes.map((scene) => {
              const isActive = scene.id === activeSceneId;
              return (
                <button
                  key={scene.id}
                  type="button"
                  onClick={() => setActiveScene(scene.id)}
                  className={[
                    "w-full text-left rounded px-3 py-2.5 border transition-all text-xs",
                    isActive
                      ? "prep-entry-selected border-[var(--gold)]/60"
                      : "bg-[#1a1008]/90 border-[#5c3e23] hover:border-[#8a6236]",
                  ].join(" ")}
                >
                  <div className="font-heading text-[13px] text-[var(--text-1)] truncate">
                    {scene.title || scene.name || "Untitled scene"}
                  </div>
                  {(scene.act || scene.type || scene.location) && (
                    <div className="text-[10px] text-[var(--text-2)] mt-0.5 truncate">
                      {[scene.act, scene.type, scene.location].filter(Boolean).join(" · ")}
                    </div>
                  )}
                </button>
              );
            })
          )}
        </div>
      </aside>

      <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden border-r border-[#3a2510]">
        {requireApiKey && (
          <div className="flex-shrink-0 px-4 py-2 border-b border-[#2a1a0a] bg-[#110b06]">
            <label className="text-[10px] text-[var(--text-2)] uppercase tracking-wide block mb-1">
              API key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full max-w-md rounded border border-[#4a3018] bg-[#1c1008] px-2 py-1 text-xs text-[var(--text-1)]"
              placeholder="Required for PATCH / voices API"
              autoComplete="off"
            />
          </div>
        )}

        {!selectedScene ? (
          <div className="flex-1 flex items-center justify-center text-sm text-[var(--text-2)] px-6 text-center">
            Select a scene from the list.
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-5">
            <header className="border-b border-[#3a2510] pb-3">
              <h1 className="font-heading text-lg text-[var(--gold)] leading-tight">
                {selectedScene.title || selectedScene.name || "Scene"}
              </h1>
              {(selectedScene.location || selectedScene.type) && (
                <p className="text-xs text-[var(--text-2)] mt-1">
                  {[selectedScene.location, selectedScene.type].filter(Boolean).join(" · ")}
                </p>
              )}
            </header>

            <section>
              <div className="flex items-center justify-between gap-2 mb-2">
                <h2 className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] font-heading">
                  Read-aloud
                </h2>
              </div>
              <textarea
                className="w-full min-h-[140px] rounded-md border border-[#4a3018] bg-[#1c1008] px-3 py-2 text-sm text-[var(--text-1)] leading-relaxed placeholder:text-[var(--text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
                placeholder="Text to read to players…"
                value={readDraft}
                onChange={(e) => setReadDraft(e.target.value)}
                onBlur={flushSceneContent}
              />
            </section>

            <section>
              <div className="flex items-center justify-between gap-2 mb-2">
                <h2 className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] font-heading">
                  NPC roster &amp; voices
                </h2>
                <button
                  type="button"
                  onClick={() => setCloneOpen(true)}
                  className="inline-flex items-center gap-1 rounded border border-[#5c3e23] px-2 py-1 text-[10px] uppercase tracking-wide text-[var(--gold)] hover:bg-[#1e1208]"
                >
                  <Mic2 size={12} />
                  Clone voice…
                </button>
              </div>
              {sceneNpcs.length === 0 ? (
                <p className="text-xs text-[var(--text-2)] italic">No NPCs linked to this scene.</p>
              ) : (
                <ul className="space-y-2">
                  {sceneNpcs.map((npc) => (
                    <li
                      key={npc.id}
                      className="flex flex-wrap items-center gap-2 rounded border border-[#3a2510] bg-[#130c06] px-3 py-2"
                    >
                      <div className="flex-1 min-w-[120px]">
                        <div className="font-heading text-sm text-[var(--text-1)]">{npc.name}</div>
                        {npc.role && (
                          <div className="text-[10px] text-[var(--text-2)]">{npc.role}</div>
                        )}
                      </div>
                      <select
                        className="flex-shrink-0 rounded border border-[#4a3018] bg-[#1c1008] px-2 py-1 text-xs text-[var(--text-1)] max-w-[200px]"
                        value={npc.voiceId || ""}
                        onChange={(e) => handleVoiceChange(npc, e.target.value)}
                      >
                        <option value="">— Voice —</option>
                        {voices.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.name}
                          </option>
                        ))}
                      </select>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section>
              <h2 className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] font-heading mb-2">
                GM notes
              </h2>
              <textarea
                className="w-full min-h-[100px] rounded-md border border-[#4a3018] bg-[#1c1008] px-3 py-2 text-sm text-[var(--text-1)] leading-relaxed placeholder:text-[var(--text-2)] focus:outline-none focus:ring-1 focus:ring-[var(--gold)]"
                placeholder="Private notes…"
                value={notesDraft}
                onChange={(e) => setNotesDraft(e.target.value)}
                onBlur={flushSceneContent}
              />
            </section>
          </div>
        )}
      </main>

      <PrepVoiceCloneModal open={cloneOpen} onClose={() => setCloneOpen(false)} authFetch={authFetch} />
    </div>
  );
}
