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
      <div className="dm-ghost-shell flex h-full min-h-[min(70vh,640px)] min-w-0 overflow-hidden gap-0 border border-[#2a1a08] rounded-lg">
        <aside
          className="flex-shrink-0 flex flex-col border-r border-[#2a1a08] overflow-y-auto bg-[#0e0a05]"
          style={{ width: "260px" }}
        >
          <div className="p-3 border-b border-[#2a1a08]">
            <div className="dm-ghost-label">Scenes</div>
            <div className="font-heading text-sm text-[#e7c27a] mt-2 opacity-60">No campaign loaded</div>
          </div>
          <div className="flex-1 p-2 space-y-2">
            <div className="dm-ghost-border dm-ghost-row px-3 py-2 flex items-center">
              <span className="dm-ghost-hint text-xs">Scene list</span>
            </div>
            <div className="dm-ghost-border dm-ghost-row px-3 py-2 flex items-center">
              <span className="dm-ghost-hint text-xs opacity-80">···</span>
            </div>
            <div className="dm-ghost-border dm-ghost-row px-3 py-2 flex items-center">
              <span className="dm-ghost-hint text-xs opacity-80">···</span>
            </div>
            <p className="text-[11px] text-[#5c4a38] px-1 pt-2 leading-relaxed">
              Import from Library to populate scenes and NPCs.
            </p>
            <Link
              to="/library"
              className="inline-block mt-2 mx-1 text-xs font-heading text-[#c9a227] hover:text-[#e7c27a] underline underline-offset-2"
            >
              Open Library →
            </Link>
          </div>
        </aside>

        <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden bg-[#0e0a05]">
          <div className="flex-shrink-0 px-4 py-3 border-b border-[#2a1a08]">
            <p className="text-sm text-[#e7c27a] font-heading">Prep</p>
            <p className="dm-ghost-hint text-xs mt-1">Select a scene to begin</p>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
            <section>
              <div className="dm-ghost-label mb-2">Read-aloud</div>
              <div className="dm-ghost-border min-h-[140px] px-3 py-3 flex items-start">
                <span className="dm-ghost-hint text-xs">Paste or edit read-aloud text here once a campaign is active.</span>
              </div>
            </section>
            <section>
              <div className="dm-ghost-label mb-2">NPC roster &amp; voices</div>
              <div className="dm-ghost-border min-h-[100px] px-3 py-3 space-y-2">
                <div className="dm-ghost-row dm-ghost-border border-[#2a2014] bg-[rgba(18,12,6,0.4)] px-2" />
                <div className="dm-ghost-row dm-ghost-border border-[#2a2014] bg-[rgba(18,12,6,0.4)] px-2" />
                <span className="dm-ghost-hint text-xs block pt-1">NPCs appear when linked to a scene.</span>
              </div>
            </section>
            <section>
              <div className="dm-ghost-label mb-2">GM notes</div>
              <div className="dm-ghost-border min-h-[100px] px-3 py-3">
                <span className="dm-ghost-hint text-xs">Private notes for the active scene.</span>
              </div>
            </section>
          </div>
        </main>
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
