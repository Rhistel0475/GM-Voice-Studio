/**
 * Import a flat backend parse result into the campaign context store.
 *
 * This is the "quick save" path — no review step, all entities are upserted
 * immediately. For reviewed imports use applyApprovedEntities instead.
 *
 * Handles both AI-parse (rich objects) and fast-parse (plain strings) shapes.
 * Preserves any existing campaigns in the store; just adds / updates this one.
 * Sets the new campaign as the active campaign so LiveBoard picks it up.
 */
import { useCampaignContextStore } from "../store/campaignContext";
import { createId } from "./utils/ids";
import type { Campaign, Npc, Scene, SceneTrigger, CodexEntry, Session } from "../types";

export interface CampaignImportResult {
  campaignId: string;
  npcCount: number;
  sceneCount: number;
  codexCount: number;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

type RawNpc =
  | string
  | { name: string; summary?: string; description?: string; role?: string; profession?: string; goals?: string[]; tags?: string[]; id?: string };

type RawScene = {
  title?: string;
  name?: string;
  summary?: string;
  description?: string;
  read_aloud?: string;
  act?: string;
  type?: string;
  atmosphere_type?: string;
  atmosphereType?: string;
  atmosphere?: string[] | string;
  location?: string;
  notes?: string;
  npcs?: string[];
  tags?: string[];
  connected_scenes?: Array<string | { id?: string; scene_id?: string; title?: string; name?: string }>;
  connectedScenes?: Array<string | { id?: string; scene_id?: string; title?: string; name?: string }>;
  triggers?: Array<{
    name?: string;
    type?: string;
    text?: string;
    action?: string | Record<string, unknown>;
    npc_id?: string;
    npc_name?: string;
    voice_id?: string;
  }>;
  narrator_voice_id?: string;
  session_id?: string;
  id?: string;
};

type RawSession = {
  id?: string;
  title?: string;
  campaign_id?: string | number;
  active_scene_id?: string;
  activeSceneId?: string;
  scene_id?: string;
  started_at?: string;
  startedAt?: string;
  status?: string;
};

type RawLocation =
  | string
  | { name: string; description?: string; tags?: string[]; id?: string };

type RawCodexEntry = {
  title: string;
  summary?: string;
  content?: string;
  type?: string;
  tags?: string[];
  id?: string;
};

function isNpcObj(v: RawNpc): v is Exclude<RawNpc, string> {
  return typeof v === "object" && v !== null && typeof (v as { name?: unknown }).name === "string";
}

function isLocObj(v: RawLocation): v is Exclude<RawLocation, string> {
  return typeof v === "object" && v !== null && typeof (v as { name?: unknown }).name === "string";
}

function normalizeConnectedSceneRefs(
  input: RawScene["connected_scenes"] | RawScene["connectedScenes"]
): string[] {
  const items = Array.isArray(input) ? input : input ? [input] : [];
  const refs: string[] = [];
  const seen = new Set<string>();

  for (const item of items) {
    const raw = typeof item === "string"
      ? item
      : item?.id || item?.scene_id || item?.title || item?.name || "";
    const ref = String(raw || "").trim();
    if (!ref) continue;
    const key = ref.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    refs.push(ref);
  }

  return refs;
}

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * Import a parse result payload directly into the campaign context store.
 *
 * Safe to call multiple times with the same payload — all upserts are idempotent
 * by ID. If the payload has no `id`, a new campaign is created each time.
 */
export function importParseResultToStore(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parseResult: Record<string, any>
): CampaignImportResult {
  const store = useCampaignContextStore.getState();
  const rawSessions: RawSession[] = Array.isArray(parseResult.sessions) ? parseResult.sessions : [];
  const activeRawSession = rawSessions.find(
    (session) => String(session?.status || "").toLowerCase() === "active"
  );
  const parsedActiveSessionId = parseResult.activeSessionId != null && parseResult.activeSessionId !== ""
    ? String(parseResult.activeSessionId)
    : parseResult.active_session_id != null && parseResult.active_session_id !== ""
      ? String(parseResult.active_session_id)
      : activeRawSession?.id != null
        ? String(activeRawSession.id)
        : undefined;

  // ── Campaign ───────────────────────────────────────────────────────────────
  const campaignId: string =
    parseResult.id != null && parseResult.id !== ""
      ? String(parseResult.id)
      : createId("campaign");
  const campaign: Campaign = {
    id: campaignId,
    name: (parseResult.title as string) || "Imported Campaign",
    setting: typeof parseResult.summary === "string"
      ? parseResult.summary.slice(0, 300)
      : undefined,
    activeSessionId: parsedActiveSessionId,
  };
  store.upsertCampaign(campaign);

  // Keep a name→id map so scene.npcs can be linked to NPC IDs.
  const npcIdByName = new Map<string, string>();
  let npcCount = 0;
  let sceneCount = 0;
  let codexCount = 0;

  // ── Sessions ───────────────────────────────────────────────────────────────
  for (const raw of rawSessions) {
    const sessionId = raw?.id != null && raw.id !== "" ? String(raw.id) : createId("session");
    const session: Session = {
      id: sessionId,
      campaignId,
      title: String(raw?.title || "Session").trim() || "Session",
      activeSceneId: raw?.activeSceneId != null && raw.activeSceneId !== ""
        ? String(raw.activeSceneId)
        : raw?.active_scene_id != null && raw.active_scene_id !== ""
          ? String(raw.active_scene_id)
          : raw?.scene_id != null && raw.scene_id !== ""
            ? String(raw.scene_id)
            : undefined,
      startedAt: raw?.startedAt || raw?.started_at || undefined,
      status: raw?.status === "active" || raw?.status === "closed" || raw?.status === "prep"
        ? raw.status
        : undefined,
    };
    store.upsertSession(session);
  }

  // ── NPCs ───────────────────────────────────────────────────────────────────
  const rawNpcs: RawNpc[] = Array.isArray(parseResult.npcs) ? parseResult.npcs : [];
  for (const raw of rawNpcs) {
    const name = (isNpcObj(raw) ? raw.name : String(raw)).trim();
    if (!name) continue;

    const npcId = (isNpcObj(raw) && raw.id != null && raw.id !== "") ? String(raw.id) : createId("npc");
    const npc: Npc = {
      id: npcId,
      campaignId,
      name,
      summary: isNpcObj(raw) ? (raw.summary || raw.description || "") : "",
      role: isNpcObj(raw) ? raw.role : undefined,
      profession: isNpcObj(raw) ? raw.profession : undefined,
      tags: isNpcObj(raw) && Array.isArray(raw.tags) ? raw.tags : [],
    };
    store.upsertNpc(npc);
    npcIdByName.set(name, npcId);
    npcCount++;
  }

  // ── Scenes ─────────────────────────────────────────────────────────────────
  const rawScenes: RawScene[] = Array.isArray(parseResult.scenes) ? parseResult.scenes : [];
  for (const raw of rawScenes) {
    const sceneTitle = String(raw?.title || raw?.name || "").trim();
    if (!sceneTitle) continue;

    const sceneId = raw.id != null && raw.id !== "" ? String(raw.id) : sceneTitle;
    const npcIds: string[] = (Array.isArray(raw.npcs) ? raw.npcs : [])
      .map((n: string) => npcIdByName.get(n.trim()))
      .filter((id): id is string => Boolean(id));
    const connectedScenes = normalizeConnectedSceneRefs(raw.connected_scenes || raw.connectedScenes);
    const triggers: SceneTrigger[] = (Array.isArray(raw.triggers) ? raw.triggers : [])
      .filter((trigger): trigger is NonNullable<typeof trigger> => Boolean(trigger?.name))
      .map((trigger) => ({
        name: String(trigger.name).trim(),
        type: String(trigger.type || "text").trim(),
        text: typeof trigger.text === "string" ? trigger.text : undefined,
        action: typeof trigger.action === "string" || (trigger.action && typeof trigger.action === "object")
          ? trigger.action
          : undefined,
        npcId: typeof trigger.npc_id === "string" ? trigger.npc_id : undefined,
        npcName: typeof trigger.npc_name === "string" ? trigger.npc_name : undefined,
        voiceId: typeof trigger.voice_id === "string" ? trigger.voice_id : undefined,
      }));

    const scene: Scene = {
      id: sceneId,
      campaignId,
      title: sceneTitle,
      name: String(raw.name || sceneTitle).trim() || sceneTitle,
      summary: raw.summary || raw.description || raw.read_aloud || "",
      description: raw.description || raw.summary || raw.read_aloud || raw.notes || "",
      act: raw.act || "",
      type: raw.type || "",
      atmosphereType: raw.atmosphereType || raw.atmosphere_type || (Array.isArray(raw.atmosphere) ? raw.atmosphere[0] : raw.atmosphere) || "",
      atmosphere_type: raw.atmosphere_type || raw.atmosphereType || (Array.isArray(raw.atmosphere) ? raw.atmosphere[0] : raw.atmosphere) || "",
      npcIds,
      codexEntryIds: [],
      actionLogIds: [],
      narrationClipIds: [],
      tags: Array.isArray(raw.tags) ? raw.tags : (raw.act ? [raw.act] : []),
      location: raw.location || "",
      notes: raw.notes || "",
      readAloud: raw.read_aloud || "",
      read_aloud: raw.read_aloud || "",
      narrator_voice_id: raw.narrator_voice_id,
      sessionId: raw.session_id,
      triggers,
      connectedScenes,
      connected_scenes: connectedScenes,
    };
    store.upsertScene(scene);
    sceneCount++;
  }

  // ── Locations → CodexEntry(type:"location") ────────────────────────────────
  const rawLocations: RawLocation[] = Array.isArray(parseResult.locations) ? parseResult.locations : [];
  for (const raw of rawLocations) {
    const title = (isLocObj(raw) ? raw.name : String(raw)).trim();
    if (!title) continue;

    const entry: CodexEntry = {
      id: (isLocObj(raw) && raw.id != null && raw.id !== "") ? String(raw.id) : createId("loc"),
      campaignId,
      type: "location",
      title,
      summary: isLocObj(raw) ? (raw.description || "") : "",
      tags: isLocObj(raw) && Array.isArray(raw.tags) ? raw.tags : [],
    };
    store.upsertCodexEntry(entry);
    codexCount++;
  }

  // ── Codex entries ──────────────────────────────────────────────────────────
  const rawCodex: RawCodexEntry[] = Array.isArray(parseResult.codex_entries) ? parseResult.codex_entries : [];
  for (const raw of rawCodex) {
    if (!raw?.title?.trim()) continue;

    const entry: CodexEntry = {
      id: raw.id != null && raw.id !== "" ? String(raw.id) : createId("codex"),
      campaignId,
      type: "lore",
      title: raw.title.trim(),
      summary: raw.summary || "",
      content: raw.content,
      tags: Array.isArray(raw.tags) ? raw.tags : [],
    };
    store.upsertCodexEntry(entry);
    codexCount++;
  }

  store.setActiveCampaign(campaignId);

  return { campaignId, npcCount, sceneCount, codexCount };
}
