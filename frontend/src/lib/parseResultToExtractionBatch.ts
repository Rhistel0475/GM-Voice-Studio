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
import { normalizeGameSystemPlugin } from "./gameSystemPlugins";

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
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_scenes?: string[];
  related_locations?: string[];
  related_factions?: string[];
  related_codex_entries?: string[];
}

interface RawSceneObject {
  title: string;
  summary?: string;
  read_aloud?: string;
  act?: string;
  npcs?: string[];
  location?: string;
  atmosphere?: string[];
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_locations?: string[];
  related_quests?: string[];
  related_codex_entries?: string[];
}

interface RawLocationObject {
  name: string;
  description?: string;
  atmosphere?: string[];
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_npcs?: string[];
  related_scenes?: string[];
  related_factions?: string[];
  related_codex_entries?: string[];
}

interface RawCodexEntryObject {
  title: string;
  summary?: string;
  content?: string;
  type?: string;
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_npcs?: string[];
  related_locations?: string[];
  related_scenes?: string[];
  related_factions?: string[];
}

interface RawQuestObject {
  id?: string;
  name?: string;
  title?: string;
  description?: string;
  summary?: string;
  objective?: string;
  stakes?: string;
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_npcs?: string[];
  related_locations?: string[];
  related_scenes?: string[];
}

interface RawFactionObject {
  id?: string;
  name?: string;
  title?: string;
  description?: string;
  summary?: string;
  goals?: string[];
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_npcs?: string[];
  related_locations?: string[];
  related_scenes?: string[];
  related_codex_entries?: string[];
}

interface RawLoreObject {
  id?: string;
  title?: string;
  name?: string;
  summary?: string;
  description?: string;
  content?: string;
  type?: string;
  tags?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
  related_npcs?: string[];
  related_locations?: string[];
  related_scenes?: string[];
  related_factions?: string[];
}

interface RawEntityMeta {
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  source?: Record<string, unknown>;
}

type RawNpc = string | RawNpcObject;
type RawLocation = string | RawLocationObject;

function isRawNpcObject(v: RawNpc): v is RawNpcObject {
  return typeof v === "object" && v !== null && typeof (v as RawNpcObject).name === "string";
}

function isRawLocationObject(v: RawLocation): v is RawLocationObject {
  return typeof v === "object" && v !== null && typeof (v as RawLocationObject).name === "string";
}

function toConfidenceLabel(value: number): ExtractionConfidence {
  if (value >= 0.82) return "high";
  if (value >= 0.58) return "medium";
  return "low";
}

function deriveReviewMeta(
  raw: RawEntityMeta | null | undefined,
  fallback: ExtractionConfidence
): { confidence: ExtractionConfidence; reviewStatus: "approved" | "needs_review"; hidden: boolean } {
  const score = typeof raw?.confidence_score === "number"
    ? raw.confidence_score
    : typeof raw?.confidence === "number"
      ? raw.confidence
      : fallback === "high"
        ? 0.85
        : 0.65;
  const confidence = raw?.confidence_label === "high" || raw?.confidence_label === "medium" || raw?.confidence_label === "low"
    ? raw.confidence_label
    : toConfidenceLabel(score);
  const priority = String(raw?.review_priority || "").trim().toLowerCase();
  const hidden = priority === "hidden" || confidence === "low";
  const reviewStatus = priority === "auto_approve" || (confidence === "high" && raw?.needs_review !== true)
    ? "approved"
    : "needs_review";
  return { confidence, reviewStatus, hidden };
}

function toRefs(values: unknown, type: "npc" | "location" | "scene" | "codex" | "faction") {
  if (!Array.isArray(values)) return [];
  return values
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .map((refName) => ({ type, refName }));
}

// ── Main export ───────────────────────────────────────────────────────────────

export function parseResultToExtractionBatch(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  parseResult: Record<string, any>,
  documentName?: string
): ExtractionBatchResult {
  const docId = createId("doc");
  const docName = documentName || (parseResult.title as string) || "Uploaded Document";
  const campaignSystem = normalizeGameSystemPlugin(
    parseResult.system || { id: parseResult.system_id || parseResult.systemId }
  );

  // Detect AI-parse (rich objects) vs fast-parse (strings) from NPC shape.
  const rawNpcs: RawNpc[] = Array.isArray(parseResult.npcs) ? parseResult.npcs : [];
  const isAiResult = rawNpcs.length > 0 && isRawNpcObject(rawNpcs[0]);
  const baseConfidence: ExtractionConfidence = isAiResult ? "high" : "medium";

  const makeSource = (
    section: string,
    excerpt?: string,
    rawSource?: Record<string, unknown>
  ): ExtractionSourceRef => ({
    sourceType: "uploaded",
    documentId: typeof rawSource?.document_id === "string" ? rawSource.document_id : docId,
    documentName: docName,
    chunkId: typeof rawSource?.chunk_id === "string" ? rawSource.chunk_id : undefined,
    pageNumber: typeof rawSource?.page_number === "number" ? rawSource.page_number : undefined,
    sectionHeading:
      typeof rawSource?.heading === "string" && rawSource.heading.trim()
        ? rawSource.heading
        : section,
    excerpt: excerpt ? excerpt.slice(0, 200) : undefined,
  });

  const entities: ExtractionEntity[] = [];
  let hiddenCount = 0;

  // ── NPCs ──────────────────────────────────────────────────────────────────
  for (const raw of rawNpcs) {
    if (isRawNpcObject(raw)) {
      const reviewMeta = deriveReviewMeta(raw, baseConfidence);
      if (reviewMeta.hidden) {
        hiddenCount++;
        continue;
      }
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
        factionRefs: toRefs(raw.related_factions, "faction"),
        locationRefs: toRefs(raw.related_locations, "location"),
        sceneRefs: toRefs(raw.related_scenes, "scene"),
        codexRefs: toRefs(raw.related_codex_entries, "codex"),
        voiceHint: raw.voiceHint || raw.voice_hint,
        tags: Array.isArray(raw.tags) ? raw.tags : [],
        confidence: reviewMeta.confidence,
        reviewStatus: reviewMeta.reviewStatus,
        source: makeSource("NPCs", raw.name, raw.source),
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
    const reviewMeta = deriveReviewMeta(raw, baseConfidence);
    if (reviewMeta.hidden) {
      hiddenCount++;
      continue;
    }
    const npcNames: string[] = Array.isArray(raw.npcs) ? raw.npcs : [];
    entities.push({
      id: createId("scene"),
      type: "scene_seed",
      title: raw.title,
      summary: raw.summary || raw.read_aloud || "",
      setupText: raw.read_aloud,
      atmosphere: Array.isArray(raw.atmosphere) ? raw.atmosphere : [],
      likelyNpcRefs: npcNames.map((n) => ({ type: "npc", refName: n })),
      likelyLocationRefs: toRefs([raw.location, ...(raw.related_locations || [])].filter(Boolean), "location"),
      likelyCodexRefs: toRefs(raw.related_codex_entries, "codex"),
      immediateHooks: [],
      possibleConflicts: Array.isArray(raw.related_quests) ? raw.related_quests : [],
      suggestedNarration: raw.read_aloud,
      tags: Array.isArray(raw.tags) ? raw.tags : (raw.act ? [raw.act] : []),
      confidence: reviewMeta.confidence,
      reviewStatus: reviewMeta.reviewStatus,
      source: makeSource("Scenes", raw.title, raw.source),
    });
  }

  // ── Locations ─────────────────────────────────────────────────────────────
  const rawLocations: RawLocation[] = Array.isArray(parseResult.locations)
    ? (parseResult.locations as RawLocation[])
    : [];
  for (const raw of rawLocations) {
    if (isRawLocationObject(raw)) {
      if (!raw.name?.trim()) continue;
      const reviewMeta = deriveReviewMeta(raw, baseConfidence);
      if (reviewMeta.hidden) {
        hiddenCount++;
        continue;
      }
      entities.push({
        id: createId("loc"),
        type: "location",
        name: raw.name.trim(),
        summary: raw.description || "",
        atmosphere: Array.isArray(raw.atmosphere) ? raw.atmosphere : [],
        notableFeatures: [],
        npcRefs: toRefs(raw.related_npcs, "npc"),
        factionRefs: toRefs(raw.related_factions, "faction"),
        sceneRefs: toRefs(raw.related_scenes, "scene"),
        codexRefs: toRefs(raw.related_codex_entries, "codex"),
        questHooks: [],
        tags: Array.isArray(raw.tags) ? raw.tags : [],
        confidence: reviewMeta.confidence,
        reviewStatus: reviewMeta.reviewStatus,
        source: makeSource("Locations", raw.name, raw.source),
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
    const reviewMeta = deriveReviewMeta(raw, baseConfidence);
    if (reviewMeta.hidden) {
      hiddenCount++;
      continue;
    }
    const entryType = raw.type === "rule" || raw.type === "faction" ? raw.type : "lore";
    entities.push({
      id: createId("codex"),
      type: "codex_entry",
      entryType,
      title: raw.title,
      summary: raw.summary || "",
      content: raw.content,
      npcRefs: toRefs(raw.related_npcs, "npc"),
      locationRefs: toRefs(raw.related_locations, "location"),
      sceneRefs: toRefs(raw.related_scenes, "scene"),
      factionRefs: toRefs(raw.related_factions, "faction"),
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidence: reviewMeta.confidence,
      reviewStatus: reviewMeta.reviewStatus,
      source: makeSource("Codex Entries", raw.title, raw.source),
    });
  }

  const rawQuests: RawQuestObject[] = Array.isArray(parseResult.quests)
    ? (parseResult.quests as RawQuestObject[])
    : [];
  for (const raw of rawQuests) {
    const title = String(raw.name || raw.title || "").trim();
    if (!title) continue;
    const reviewMeta = deriveReviewMeta(raw, baseConfidence);
    if (reviewMeta.hidden) {
      hiddenCount++;
      continue;
    }
    entities.push({
      id: raw.id || createId("quest"),
      type: "codex_entry",
      entryType: "quest_note",
      title,
      summary: raw.description || raw.summary || raw.objective || "",
      content: [raw.objective, raw.stakes].filter(Boolean).join("\n"),
      npcRefs: toRefs(raw.related_npcs, "npc"),
      locationRefs: toRefs(raw.related_locations, "location"),
      sceneRefs: toRefs(raw.related_scenes, "scene"),
      factionRefs: [],
      tags: Array.isArray(raw.tags) && raw.tags.length > 0 ? raw.tags : ["quest"],
      confidence: reviewMeta.confidence,
      reviewStatus: reviewMeta.reviewStatus,
      source: makeSource("Quests", title, raw.source),
    });
  }

  const rawFactions: RawFactionObject[] = Array.isArray(parseResult.factions)
    ? (parseResult.factions as RawFactionObject[])
    : [];
  for (const raw of rawFactions) {
    const name = String(raw.name || raw.title || "").trim();
    if (!name) continue;
    const reviewMeta = deriveReviewMeta(raw, baseConfidence);
    if (reviewMeta.hidden) {
      hiddenCount++;
      continue;
    }
    entities.push({
      id: raw.id || createId("faction"),
      type: "faction",
      name,
      description: raw.description || raw.summary || "",
      goals: Array.isArray(raw.goals) ? raw.goals : [],
      knownMemberRefs: toRefs(raw.related_npcs, "npc"),
      alliedNpcRefs: [],
      enemyFactionRefs: [],
      locationRefs: toRefs(raw.related_locations, "location"),
      sceneRefs: toRefs(raw.related_scenes, "scene"),
      codexRefs: toRefs(raw.related_codex_entries, "codex"),
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidence: reviewMeta.confidence,
      reviewStatus: reviewMeta.reviewStatus,
      source: makeSource("Factions", name, raw.source),
    });
  }

  const rawLore: RawLoreObject[] = Array.isArray(parseResult.lore)
    ? (parseResult.lore as RawLoreObject[])
    : [];
  for (const raw of rawLore) {
    const title = String(raw.title || raw.name || "").trim();
    if (!title) continue;
    const reviewMeta = deriveReviewMeta(raw, baseConfidence);
    if (reviewMeta.hidden) {
      hiddenCount++;
      continue;
    }
    entities.push({
      id: raw.id || createId("lore"),
      type: "codex_entry",
      entryType: raw.type === "rule" ? "rule" : "lore",
      title,
      summary: raw.summary || raw.description || "",
      content: raw.content,
      npcRefs: toRefs(raw.related_npcs, "npc"),
      locationRefs: toRefs(raw.related_locations, "location"),
      sceneRefs: toRefs(raw.related_scenes, "scene"),
      factionRefs: toRefs(raw.related_factions, "faction"),
      tags: Array.isArray(raw.tags) ? raw.tags : [],
      confidence: reviewMeta.confidence,
      reviewStatus: reviewMeta.reviewStatus,
      source: makeSource("Lore", title, raw.source),
    });
  }

  return {
    entities,
    documentId: docId,
    documentName: docName,
    extractedAt: nowIso(),
    notes: [
      ...(campaignSystem ? [`Campaign system: ${campaignSystem.label}`] : []),
      ...(hiddenCount > 0 ? [`${hiddenCount} low-confidence entities were hidden from the review queue.`] : []),
    ],
  };
}
