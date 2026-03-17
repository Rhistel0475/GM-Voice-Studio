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
import {
  loadGameSystemPlugin,
  normalizeGameSystemId,
  normalizeGameSystemPlugin,
} from "./gameSystemPlugins";

export interface CampaignImportResult {
  campaignId: string;
  npcCount: number;
  sceneCount: number;
  codexCount: number;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

type RawNpc =
  | string
  | {
      name: string;
      summary?: string;
      description?: string;
      role?: string;
      profession?: string;
      faction?: string;
      goals?: string[];
      tags?: string[];
      id?: string;
      confidence?: number;
      confidence_score?: number;
      confidence_label?: string;
      needs_review?: boolean;
      related_scenes?: string[];
      related_locations?: string[];
    };

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
  ambience_track?: string | { url?: string; label?: string; atmosphere_type?: string; filename?: string; loop?: boolean; volume?: number; track_id?: string } | null;
  ambienceTrack?: string | { url?: string; label?: string; atmosphere_type?: string; filename?: string; loop?: boolean; volume?: number; track_id?: string } | null;
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
  | {
      name: string;
      description?: string;
      tags?: string[];
      id?: string;
      confidence?: number;
      confidence_score?: number;
      confidence_label?: string;
      needs_review?: boolean;
      related_npcs?: string[];
      related_scenes?: string[];
      related_factions?: string[];
    };

type RawCodexEntry = {
  title: string;
  summary?: string;
  content?: string;
  type?: string;
  tags?: string[];
  id?: string;
};

type RawKnowledgeRecord = {
  id?: string;
  name?: string;
  title?: string;
  description?: string;
  summary?: string;
  content?: string;
  objective?: string;
  stakes?: string;
  tags?: string[];
  type?: string;
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

function normalizeKnowledgeRecords(
  value: unknown
): Campaign["quests"] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is RawKnowledgeRecord => Boolean(item && typeof item === "object" && !Array.isArray(item)))
    .map((item) => ({
      ...item,
      id: item.id != null && item.id !== "" ? String(item.id) : undefined,
      name: typeof item.name === "string" ? item.name : undefined,
      title: typeof item.title === "string" ? item.title : undefined,
      description: typeof item.description === "string" ? item.description : undefined,
      summary: typeof item.summary === "string" ? item.summary : undefined,
      content: typeof item.content === "string" ? item.content : undefined,
      tags: Array.isArray(item.tags) ? item.tags.map((tag) => String(tag)) : [],
    }));
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
  const systemId = normalizeGameSystemId(parseResult.system_id || parseResult.systemId || parseResult.system?.id);
  const system = normalizeGameSystemPlugin(parseResult.system) || loadGameSystemPlugin(systemId);
  const campaign: Campaign = {
    id: campaignId,
    name: (parseResult.title as string) || "Imported Campaign",
    setting: typeof parseResult.summary === "string"
      ? parseResult.summary.slice(0, 300)
      : undefined,
    activeSessionId: parsedActiveSessionId,
    systemId,
    system,
    quests: normalizeKnowledgeRecords(parseResult.quests),
    factions: normalizeKnowledgeRecords(parseResult.factions),
    lore: normalizeKnowledgeRecords(parseResult.lore),
    relationships: Array.isArray(parseResult.relationships) ? parseResult.relationships : [],
    reviewSummary: parseResult.review_summary && typeof parseResult.review_summary === "object"
      ? parseResult.review_summary
      : undefined,
  };
  store.upsertCampaign(campaign);

  // Keep a name→id map so scene.npcs can be linked to NPC IDs.
  const npcIdByName = new Map<string, string>();
  const locationIdByName = new Map<string, string>();
  const sceneIdByName = new Map<string, string>();
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
      description: isNpcObj(raw) ? (raw.description || "") : undefined,
      personality: isNpcObj(raw) ? (raw.summary || raw.description || "") : undefined,
      role: isNpcObj(raw) ? raw.role : undefined,
      profession: isNpcObj(raw) ? raw.profession : undefined,
      factionName: isNpcObj(raw) ? (typeof raw.faction === "string" ? raw.faction : undefined) : undefined,
      tags: isNpcObj(raw) && Array.isArray(raw.tags) ? raw.tags : [],
      relatedSceneNames: isNpcObj(raw) && Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? (raw as Record<string, string[]>).related_scenes
        : [],
      relatedLocationNames: isNpcObj(raw) && Array.isArray((raw as Record<string, unknown>).related_locations)
        ? (raw as Record<string, string[]>).related_locations
        : [],
      confidenceScore: isNpcObj(raw) && typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : isNpcObj(raw) && typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: isNpcObj(raw) && typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as Npc["confidenceLabel"])
        : undefined,
      needsReview: isNpcObj(raw) && (raw as Record<string, unknown>).needs_review === true,
    };
    store.upsertNpc(npc);
    npcIdByName.set(name, npcId);
    npcCount++;
  }

  const rawLocations: RawLocation[] = Array.isArray(parseResult.locations) ? parseResult.locations : [];
  for (const raw of rawLocations) {
    const title = (isLocObj(raw) ? raw.name : String(raw)).trim();
    if (!title) continue;
    const locationId = (isLocObj(raw) && raw.id != null && raw.id !== "") ? String(raw.id) : createId("loc");
    locationIdByName.set(title, locationId);
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
      locationId: raw.location ? locationIdByName.get(String(raw.location).trim()) : undefined,
      npcIds,
      codexEntryIds: raw.location ? [locationIdByName.get(String(raw.location).trim())].filter(Boolean) as string[] : [],
      actionLogIds: [],
      narrationClipIds: [],
      tags: Array.isArray(raw.tags) ? raw.tags : (raw.act ? [raw.act] : []),
      location: raw.location || "",
      relatedNpcNames: Array.isArray((raw as Record<string, unknown>).related_npcs) ? (raw as Record<string, string[]>).related_npcs : [],
      relatedLocationNames: Array.isArray((raw as Record<string, unknown>).related_locations) ? (raw as Record<string, string[]>).related_locations : [],
      relatedQuestNames: Array.isArray((raw as Record<string, unknown>).related_quests) ? (raw as Record<string, string[]>).related_quests : [],
      relatedFactionNames: Array.isArray((raw as Record<string, unknown>).related_factions) ? (raw as Record<string, string[]>).related_factions : [],
      confidenceScore: typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as Scene["confidenceLabel"])
        : undefined,
      needsReview: (raw as Record<string, unknown>).needs_review === true,
      notes: raw.notes || "",
      readAloud: raw.read_aloud || "",
      read_aloud: raw.read_aloud || "",
      ambienceTrack: raw.ambienceTrack || raw.ambience_track || null,
      ambience_track: raw.ambience_track || raw.ambienceTrack || null,
      narrator_voice_id: raw.narrator_voice_id,
      sessionId: raw.session_id,
      triggers,
      connectedScenes,
      connected_scenes: connectedScenes,
    };
    store.upsertScene(scene);
    sceneIdByName.set(sceneTitle, sceneId);
    sceneCount++;
  }

  // ── Locations → CodexEntry(type:"location") ────────────────────────────────
  for (const raw of rawLocations) {
    const title = (isLocObj(raw) ? raw.name : String(raw)).trim();
    if (!title) continue;

    const entry: CodexEntry = {
      id: locationIdByName.get(title) || createId("loc"),
      campaignId,
      type: "location",
      title,
      summary: isLocObj(raw) ? (raw.description || "") : "",
      relatedNpcNames: isLocObj(raw) && Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? (raw as Record<string, string[]>).related_npcs
        : [],
      relatedSceneNames: isLocObj(raw) && Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? (raw as Record<string, string[]>).related_scenes
        : [],
      relatedFactionNames: isLocObj(raw) && Array.isArray((raw as Record<string, unknown>).related_factions)
        ? (raw as Record<string, string[]>).related_factions
        : [],
      relatedNpcIds: isLocObj(raw) && Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? ((raw as Record<string, string[]>).related_npcs || []).map((name) => npcIdByName.get(name)).filter(Boolean) as string[]
        : [],
      relatedSceneIds: isLocObj(raw) && Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? ((raw as Record<string, string[]>).related_scenes || []).map((name) => sceneIdByName.get(name)).filter(Boolean) as string[]
        : [],
      tags: isLocObj(raw) && Array.isArray(raw.tags) ? raw.tags : [],
      confidenceScore: isLocObj(raw) && typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : isLocObj(raw) && typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: isLocObj(raw) && typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as CodexEntry["confidenceLabel"])
        : undefined,
      needsReview: isLocObj(raw) && (raw as Record<string, unknown>).needs_review === true,
    };
    store.upsertCodexEntry(entry);
    codexCount++;
  }

  // ── Codex entries ──────────────────────────────────────────────────────────
  const rawCodex: RawCodexEntry[] = Array.isArray(parseResult.codex_entries) ? parseResult.codex_entries : [];
  for (const raw of rawCodex) {
    if (!raw?.title?.trim()) continue;
    const normalizedType = raw.type === "rule" || raw.type === "faction" ? raw.type : "lore";

    const entry: CodexEntry = {
      id: raw.id != null && raw.id !== "" ? String(raw.id) : createId("codex"),
      campaignId,
      type: normalizedType,
      title: raw.title.trim(),
      summary: raw.summary || "",
      content: raw.content,
      relatedNpcNames: Array.isArray((raw as Record<string, unknown>).related_npcs) ? (raw as Record<string, string[]>).related_npcs : [],
      relatedLocationNames: Array.isArray((raw as Record<string, unknown>).related_locations) ? (raw as Record<string, string[]>).related_locations : [],
      relatedSceneNames: Array.isArray((raw as Record<string, unknown>).related_scenes) ? (raw as Record<string, string[]>).related_scenes : [],
      relatedFactionNames: Array.isArray((raw as Record<string, unknown>).related_factions) ? (raw as Record<string, string[]>).related_factions : [],
      relatedNpcIds: Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? ((raw as Record<string, string[]>).related_npcs || []).map((name) => npcIdByName.get(name)).filter(Boolean) as string[]
        : [],
      relatedSceneIds: Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? ((raw as Record<string, string[]>).related_scenes || []).map((name) => sceneIdByName.get(name)).filter(Boolean) as string[]
        : [],
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidenceScore: typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as CodexEntry["confidenceLabel"])
        : undefined,
      needsReview: (raw as Record<string, unknown>).needs_review === true,
    };
    store.upsertCodexEntry(entry);
    codexCount++;
  }

  const rawFactions: RawKnowledgeRecord[] = Array.isArray(parseResult.factions) ? parseResult.factions : [];
  for (const raw of rawFactions) {
    const title = String(raw.name || raw.title || "").trim();
    if (!title) continue;
    store.upsertCodexEntry({
      id: raw.id != null && raw.id !== "" ? String(raw.id) : createId("faction"),
      campaignId,
      type: "faction",
      title,
      summary: raw.description || raw.summary || "",
      content: raw.content,
      relatedNpcNames: Array.isArray((raw as Record<string, unknown>).related_npcs) ? (raw as Record<string, string[]>).related_npcs : [],
      relatedLocationNames: Array.isArray((raw as Record<string, unknown>).related_locations) ? (raw as Record<string, string[]>).related_locations : [],
      relatedSceneNames: Array.isArray((raw as Record<string, unknown>).related_scenes) ? (raw as Record<string, string[]>).related_scenes : [],
      relatedNpcIds: Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? ((raw as Record<string, string[]>).related_npcs || []).map((name) => npcIdByName.get(name)).filter(Boolean) as string[]
        : [],
      relatedSceneIds: Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? ((raw as Record<string, string[]>).related_scenes || []).map((name) => sceneIdByName.get(name)).filter(Boolean) as string[]
        : [],
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidenceScore: typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as CodexEntry["confidenceLabel"])
        : undefined,
      needsReview: (raw as Record<string, unknown>).needs_review === true,
    });
    codexCount++;
  }

  const rawLore: RawKnowledgeRecord[] = Array.isArray(parseResult.lore) ? parseResult.lore : [];
  for (const raw of rawLore) {
    const title = String(raw.title || raw.name || "").trim();
    if (!title) continue;
    store.upsertCodexEntry({
      id: raw.id != null && raw.id !== "" ? String(raw.id) : createId("lore"),
      campaignId,
      type: raw.type === "rule" ? "rule" : "lore",
      title,
      summary: raw.summary || raw.description || "",
      content: raw.content,
      relatedNpcNames: Array.isArray((raw as Record<string, unknown>).related_npcs) ? (raw as Record<string, string[]>).related_npcs : [],
      relatedLocationNames: Array.isArray((raw as Record<string, unknown>).related_locations) ? (raw as Record<string, string[]>).related_locations : [],
      relatedSceneNames: Array.isArray((raw as Record<string, unknown>).related_scenes) ? (raw as Record<string, string[]>).related_scenes : [],
      relatedFactionNames: Array.isArray((raw as Record<string, unknown>).related_factions) ? (raw as Record<string, string[]>).related_factions : [],
      relatedNpcIds: Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? ((raw as Record<string, string[]>).related_npcs || []).map((name) => npcIdByName.get(name)).filter(Boolean) as string[]
        : [],
      relatedSceneIds: Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? ((raw as Record<string, string[]>).related_scenes || []).map((name) => sceneIdByName.get(name)).filter(Boolean) as string[]
        : [],
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidenceScore: typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as CodexEntry["confidenceLabel"])
        : undefined,
      needsReview: (raw as Record<string, unknown>).needs_review === true,
    });
    codexCount++;
  }

  const rawQuests: RawKnowledgeRecord[] = Array.isArray(parseResult.quests) ? parseResult.quests : [];
  for (const raw of rawQuests) {
    const title = String(raw.name || raw.title || "").trim();
    if (!title) continue;
    const summary = String(raw.description || raw.summary || raw.objective || "").trim();
    const questTags = new Set<string>(["quest"]);
    if (Array.isArray(raw.tags)) {
      raw.tags.forEach((tag) => {
        const normalized = String(tag || "").trim();
        if (normalized) questTags.add(normalized);
      });
    }
    store.upsertCodexEntry({
      id: raw.id != null && raw.id !== "" ? String(raw.id) : createId("quest"),
      campaignId,
      type: "lore",
      title,
      summary,
      content: [raw.objective, raw.stakes].filter(Boolean).join("\n"),
      relatedNpcNames: Array.isArray((raw as Record<string, unknown>).related_npcs) ? (raw as Record<string, string[]>).related_npcs : [],
      relatedLocationNames: Array.isArray((raw as Record<string, unknown>).related_locations) ? (raw as Record<string, string[]>).related_locations : [],
      relatedSceneNames: Array.isArray((raw as Record<string, unknown>).related_scenes) ? (raw as Record<string, string[]>).related_scenes : [],
      relatedQuestNames: [title],
      relatedNpcIds: Array.isArray((raw as Record<string, unknown>).related_npcs)
        ? ((raw as Record<string, string[]>).related_npcs || []).map((name) => npcIdByName.get(name)).filter(Boolean) as string[]
        : [],
      relatedSceneIds: Array.isArray((raw as Record<string, unknown>).related_scenes)
        ? ((raw as Record<string, string[]>).related_scenes || []).map((name) => sceneIdByName.get(name)).filter(Boolean) as string[]
        : [],
      tags: Array.from(questTags),
      confidenceScore: typeof (raw as Record<string, unknown>).confidence_score === "number"
        ? Number((raw as Record<string, number>).confidence_score)
        : typeof (raw as Record<string, unknown>).confidence === "number"
          ? Number((raw as Record<string, number>).confidence)
          : undefined,
      confidenceLabel: typeof (raw as Record<string, unknown>).confidence_label === "string"
        ? ((raw as Record<string, string>).confidence_label as CodexEntry["confidenceLabel"])
        : undefined,
      needsReview: (raw as Record<string, unknown>).needs_review === true,
    });
    codexCount++;
  }

  store.setActiveCampaign(campaignId);

  return { campaignId, npcCount, sceneCount, codexCount };
}
