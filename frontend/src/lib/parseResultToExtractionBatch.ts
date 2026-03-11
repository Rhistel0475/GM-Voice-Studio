/**
 * Maps a backend parse result (flat campaign payload) to an ExtractionBatchResult
 * suitable for enqueueing into the extraction review queue.
 *
 * Handles both AI-parse (rich objects) and fast-parse (plain strings) output shapes.
 */
import type {
  ExtractionBatchResult,
  ExtractionEntity,
  ExtractionSourceRef,
  ExtractionConfidence,
} from "../types";
import { createId, nowIso } from "./utils/ids";

// ── Raw shapes returned by the backend parse endpoints ────────────────────────

interface RawNpcObject {
  name: string;
  summary?: string;
  description?: string;
  role?: string;
  profession?: string;
  personality?: string;
  goals?: string[];
  secrets?: string[];
  tags?: string[];
  voiceHint?: string;
  voice_hint?: string;
}

interface RawSceneObject {
  title: string;
  summary?: string;
  read_aloud?: string;
  act?: string;
  npcs?: string[];
  atmosphere?: string[];
  tags?: string[];
}

interface RawLocationObject {
  name: string;
  description?: string;
  atmosphere?: string[];
  tags?: string[];
}

interface RawCodexEntryObject {
  title: string;
  summary?: string;
  content?: string;
  type?: string;
  tags?: string[];
}

type RawNpc = string | RawNpcObject;
type RawLocation = string | RawLocationObject;

function isRawNpcObject(v: RawNpc): v is RawNpcObject {
  return typeof v === "object" && v !== null && typeof (v as RawNpcObject).name === "string";
}

function isRawLocationObject(v: RawLocation): v is RawLocationObject {
  return typeof v === "object" && v !== null && typeof (v as RawLocationObject).name === "string";
}

// ── Main export ───────────────────────────────────────────────────────────────

export function parseResultToExtractionBatch(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parseResult: Record<string, any>,
  documentName?: string
): ExtractionBatchResult {
  const docId = createId("doc");
  const docName = documentName || (parseResult.title as string) || "Uploaded Document";

  // Detect AI-parse (rich objects) vs fast-parse (strings) from NPC shape.
  const rawNpcs: RawNpc[] = Array.isArray(parseResult.npcs) ? parseResult.npcs : [];
  const isAiResult = rawNpcs.length > 0 && isRawNpcObject(rawNpcs[0]);
  const baseConfidence: ExtractionConfidence = isAiResult ? "high" : "medium";

  const makeSource = (section: string, excerpt?: string): ExtractionSourceRef => ({
    sourceType: "uploaded",
    documentId: docId,
    documentName: docName,
    sectionHeading: section,
    excerpt: excerpt ? excerpt.slice(0, 200) : undefined,
  });

  const entities: ExtractionEntity[] = [];

  // ── NPCs ──────────────────────────────────────────────────────────────────
  for (const raw of rawNpcs) {
    if (isRawNpcObject(raw)) {
      entities.push({
        id: createId("npc"),
        type: "npc",
        name: raw.name,
        summary: raw.summary || raw.description || "",
        role: raw.role,
        profession: raw.profession,
        personalityTraits: raw.personality ? [raw.personality] : [],
        goals: Array.isArray(raw.goals) ? raw.goals : [],
        secrets: Array.isArray(raw.secrets) ? raw.secrets : [],
        quirks: [],
        factionRefs: [],
        locationRefs: [],
        sceneRefs: [],
        codexRefs: [],
        voiceHint: raw.voiceHint || raw.voice_hint,
        tags: Array.isArray(raw.tags) ? raw.tags : [],
        confidence: baseConfidence,
        reviewStatus: "pending",
        source: makeSource("NPCs", raw.name),
      });
    } else if (typeof raw === "string" && raw.trim()) {
      entities.push({
        id: createId("npc"),
        type: "npc",
        name: raw.trim(),
        summary: "",
        personalityTraits: [],
        goals: [],
        secrets: [],
        quirks: [],
        factionRefs: [],
        locationRefs: [],
        sceneRefs: [],
        codexRefs: [],
        tags: [],
        confidence: "medium",
        reviewStatus: "needs_review",
        source: makeSource("NPCs", raw.trim()),
      });
    }
  }

  // ── Scenes → scene seeds ──────────────────────────────────────────────────
  const rawScenes: RawSceneObject[] = Array.isArray(parseResult.scenes)
    ? (parseResult.scenes as RawSceneObject[])
    : [];
  for (const raw of rawScenes) {
    if (!raw?.title) continue;
    const npcNames: string[] = Array.isArray(raw.npcs) ? raw.npcs : [];
    entities.push({
      id: createId("scene"),
      type: "scene_seed",
      title: raw.title,
      summary: raw.summary || raw.read_aloud || "",
      setupText: raw.read_aloud,
      atmosphere: Array.isArray(raw.atmosphere) ? raw.atmosphere : [],
      likelyNpcRefs: npcNames.map((n) => ({ type: "npc", refName: n })),
      likelyLocationRefs: [],
      likelyCodexRefs: [],
      immediateHooks: [],
      possibleConflicts: [],
      suggestedNarration: raw.read_aloud,
      tags: Array.isArray(raw.tags) ? raw.tags : (raw.act ? [raw.act] : []),
      confidence: baseConfidence,
      reviewStatus: "pending",
      source: makeSource("Scenes", raw.title),
    });
  }

  // ── Locations ─────────────────────────────────────────────────────────────
  const rawLocations: RawLocation[] = Array.isArray(parseResult.locations)
    ? (parseResult.locations as RawLocation[])
    : [];
  for (const raw of rawLocations) {
    if (isRawLocationObject(raw)) {
      if (!raw.name?.trim()) continue;
      entities.push({
        id: createId("loc"),
        type: "location",
        name: raw.name.trim(),
        summary: raw.description || "",
        atmosphere: Array.isArray(raw.atmosphere) ? raw.atmosphere : [],
        notableFeatures: [],
        npcRefs: [],
        factionRefs: [],
        sceneRefs: [],
        codexRefs: [],
        questHooks: [],
        tags: Array.isArray(raw.tags) ? raw.tags : [],
        confidence: baseConfidence,
        reviewStatus: "pending",
        source: makeSource("Locations", raw.name),
      });
    } else if (typeof raw === "string" && raw.trim()) {
      entities.push({
        id: createId("loc"),
        type: "location",
        name: raw.trim(),
        summary: "",
        atmosphere: [],
        notableFeatures: [],
        npcRefs: [],
        factionRefs: [],
        sceneRefs: [],
        codexRefs: [],
        questHooks: [],
        tags: [],
        confidence: "medium",
        reviewStatus: "needs_review",
        source: makeSource("Locations", raw.trim()),
      });
    }
  }

  // ── Codex entries (if backend returned them) ──────────────────────────────
  const rawCodex: RawCodexEntryObject[] = Array.isArray(parseResult.codex_entries)
    ? (parseResult.codex_entries as RawCodexEntryObject[])
    : [];
  for (const raw of rawCodex) {
    if (!raw?.title) continue;
    entities.push({
      id: createId("codex"),
      type: "codex_entry",
      entryType: "lore",
      title: raw.title,
      summary: raw.summary || "",
      content: raw.content,
      npcRefs: [],
      locationRefs: [],
      sceneRefs: [],
      factionRefs: [],
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidence: baseConfidence,
      reviewStatus: "pending",
      source: makeSource("Codex Entries", raw.title),
    });
  }

  return {
    entities,
    documentId: docId,
    documentName: docName,
    extractedAt: nowIso(),
  };
}
