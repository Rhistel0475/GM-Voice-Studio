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
import type { Campaign, Npc, Scene, CodexEntry } from "../types";

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
  title: string;
  summary?: string;
  read_aloud?: string;
  act?: string;
  npcs?: string[];
  tags?: string[];
  id?: string;
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

  // ── Campaign ───────────────────────────────────────────────────────────────
  const campaignId: string = (parseResult.id as string) || createId("campaign");
  const campaign: Campaign = {
    id: campaignId,
    name: (parseResult.title as string) || "Imported Campaign",
    setting: typeof parseResult.summary === "string"
      ? parseResult.summary.slice(0, 300)
      : undefined,
  };
  store.upsertCampaign(campaign);
  store.setActiveCampaign(campaignId);

  // Keep a name→id map so scene.npcs can be linked to NPC IDs.
  const npcIdByName = new Map<string, string>();
  let npcCount = 0;
  let sceneCount = 0;
  let codexCount = 0;

  // ── NPCs ───────────────────────────────────────────────────────────────────
  const rawNpcs: RawNpc[] = Array.isArray(parseResult.npcs) ? parseResult.npcs : [];
  for (const raw of rawNpcs) {
    const name = (isNpcObj(raw) ? raw.name : String(raw)).trim();
    if (!name) continue;

    const npcId = (isNpcObj(raw) && raw.id) ? raw.id : createId("npc");
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
    if (!raw?.title?.trim()) continue;

    const sceneId = raw.id || createId("scene");
    const npcIds: string[] = (Array.isArray(raw.npcs) ? raw.npcs : [])
      .map((n: string) => npcIdByName.get(n.trim()))
      .filter((id): id is string => Boolean(id));

    const scene: Scene = {
      id: sceneId,
      campaignId,
      title: raw.title.trim(),
      summary: raw.summary || raw.read_aloud || "",
      npcIds,
      codexEntryIds: [],
      actionLogIds: [],
      narrationClipIds: [],
      tags: Array.isArray(raw.tags) ? raw.tags : (raw.act ? [raw.act] : []),
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
      id: (isLocObj(raw) && raw.id) ? raw.id : createId("loc"),
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
      id: raw.id || createId("codex"),
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

  return { campaignId, npcCount, sceneCount, codexCount };
}
