/**
 * Extraction schemas: canonical structured output for the document ingestion pipeline.
 * Feeds Live Board, Codex, NPC Workshop, and Voice Studio. Map to app entities when persisting.
 */

// --- Shared supporting types ---

export type ExtractionConfidence = "high" | "medium" | "low";

export type ExtractionSourceType = "placeholder" | "demo" | "uploaded" | "generated";

export type ExtractionReviewStatus = "pending" | "approved" | "rejected" | "needs_review";

export interface ExtractionSourceRef {
  sourceType: ExtractionSourceType;
  documentId?: string;
  documentName?: string;
  chunkId?: string;
  pageNumber?: number;
  sectionHeading?: string;
  excerpt?: string;
}

export type ExtractionRelationType =
  | "npc"
  | "location"
  | "scene"
  | "codex"
  | "faction"
  | "document";

export interface ExtractionRelationRef {
  type: ExtractionRelationType;
  refName: string;
  refId?: string;
}

/** Present on every extracted entity. Ensures confidence, source tracking, and review status are consistent across all types. */
export interface ExtractionMeta {
  confidence: ExtractionConfidence;
  reviewStatus: ExtractionReviewStatus;
  source: ExtractionSourceRef;
}

// --- Entity schemas ---

/** NPC extraction. Feeds: NPC Workshop, Codex, Live Board (refs), Voice Studio (voiceHint). */
export interface ExtractedNPC extends ExtractionMeta {
  id: string;
  type: "npc";
  name: string;
  summary: string;
  role?: string;
  profession?: string;
  personalityTraits: string[];
  goals: string[];
  secrets: string[];
  quirks: string[];
  factionRefs: ExtractionRelationRef[];
  locationRefs: ExtractionRelationRef[];
  sceneRefs: ExtractionRelationRef[];
  codexRefs: ExtractionRelationRef[];
  voiceHint?: string;
  tags: string[];
}

/** Location extraction. Feeds: Codex, Live Board (scene setup, quest hooks). */
export interface ExtractedLocation extends ExtractionMeta {
  id: string;
  type: "location";
  name: string;
  summary: string;
  atmosphere: string[];
  notableFeatures: string[];
  npcRefs: ExtractionRelationRef[];
  factionRefs: ExtractionRelationRef[];
  sceneRefs: ExtractionRelationRef[];
  codexRefs: ExtractionRelationRef[];
  questHooks: string[];
  sceneSeedHint?: string;
  tags: string[];
}

/** Scene seed extraction. Feeds: Live Board (session prep, narration context, scene startup). Ref arrays are suggested links for the scene. */
export interface ExtractedSceneSeed extends ExtractionMeta {
  id: string;
  type: "scene_seed";
  title: string;
  summary: string;
  setupText?: string;
  atmosphere: string[];
  likelyNpcRefs: ExtractionRelationRef[];
  likelyLocationRefs: ExtractionRelationRef[];
  likelyCodexRefs: ExtractionRelationRef[];
  immediateHooks: string[];
  possibleConflicts: string[];
  suggestedNarration?: string;
  tags: string[];
}

export type ExtractedCodexEntryType =
  | "document"
  | "lore"
  | "rule"
  | "faction"
  | "item"
  | "quest_note"
  | "reference";

/** Codex entry extraction. Feeds: Codex (research, reference lookup, scene linking). */
export interface ExtractedCodexEntry extends ExtractionMeta {
  id: string;
  type: "codex_entry";
  entryType: ExtractedCodexEntryType;
  title: string;
  summary: string;
  content?: string;
  npcRefs: ExtractionRelationRef[];
  locationRefs: ExtractionRelationRef[];
  sceneRefs: ExtractionRelationRef[];
  factionRefs: ExtractionRelationRef[];
  tags: string[];
}

/** Encounter extraction. Linkable to scenes, locations, NPCs. */
export interface ExtractedEncounter extends ExtractionMeta {
  id: string;
  type: "encounter";
  name: string;
  summary?: string;
  locationRefs: ExtractionRelationRef[];
  enemyNpcRefs: ExtractionRelationRef[];
  difficultyHint?: string;
  narrativeSetup?: string;
  treasureOrRewards: string[];
  sceneRefs: ExtractionRelationRef[];
  tags: string[];
}

/** Item/loot extraction. Codex, rewards, scene references. */
export interface ExtractedItem extends ExtractionMeta {
  id: string;
  type: "item";
  name: string;
  description?: string;
  rarityHint?: string;
  locationRefs: ExtractionRelationRef[];
  ownerRefs: ExtractionRelationRef[];
  narrativeImportance?: string;
  tags: string[];
}

/** Faction/organization extraction. Links to NPCs, locations, scenes, codex. */
export interface ExtractedFaction extends ExtractionMeta {
  id: string;
  type: "faction";
  name: string;
  description?: string;
  goals: string[];
  knownMemberRefs: ExtractionRelationRef[];
  alliedNpcRefs: ExtractionRelationRef[];
  enemyFactionRefs: ExtractionRelationRef[];
  locationRefs: ExtractionRelationRef[];
  sceneRefs: ExtractionRelationRef[];
  codexRefs: ExtractionRelationRef[];
  tags: string[];
}

// --- Union and wrapper types ---

/** Discriminated union of all extraction entity types. Use type guards or e.type to narrow. */
export type ExtractionEntity =
  | ExtractedNPC
  | ExtractedLocation
  | ExtractedSceneSeed
  | ExtractedCodexEntry
  | ExtractedEncounter
  | ExtractedItem
  | ExtractedFaction;

/** Literal type of entity discriminants for exhaustive switch. */
export type ExtractionEntityType = ExtractionEntity["type"];

export interface ExtractionBatchResult {
  entities: ExtractionEntity[];
  documentId?: string;
  documentName?: string;
  /** When this batch was produced (ISO 8601 string). */
  extractedAt?: string;
  notes?: string[];
}

/** Single review queue item wrapping an extracted entity. */
export interface ExtractionReviewItem {
  id: string;
  /** The underlying extracted entity under review. */
  entity: ExtractionEntity;
  /** Snapshot of confidence at review time. */
  confidence: ExtractionConfidence;
  /** Current review status for this item. */
  reviewStatus: ExtractionReviewStatus;
  /** Source metadata used for display (excerpt, document, chunk, etc.). */
  source: ExtractionSourceRef;
  /** When this item was enqueued into the review queue (ISO 8601 string). */
  createdAt: string;
  /** Whether this item was auto-approved by confidence rules. */
  autoApproved?: boolean;
}

/** Counts by entity type plus low-confidence and needs-review totals for pipeline/UI. */
export interface ExtractionSummary {
  npcCount: number;
  locationCount: number;
  sceneSeedCount: number;
  codexEntryCount: number;
  encounterCount: number;
  itemCount: number;
  factionCount: number;
  lowConfidenceCount: number;
  needsReviewCount: number;
}

// --- Type guards ---

export function isExtractedNPC(e: ExtractionEntity): e is ExtractedNPC {
  return e.type === "npc";
}

export function isExtractedLocation(e: ExtractionEntity): e is ExtractedLocation {
  return e.type === "location";
}

export function isExtractedSceneSeed(e: ExtractionEntity): e is ExtractedSceneSeed {
  return e.type === "scene_seed";
}

export function isExtractedCodexEntry(e: ExtractionEntity): e is ExtractedCodexEntry {
  return e.type === "codex_entry";
}

export function isExtractedEncounter(e: ExtractionEntity): e is ExtractedEncounter {
  return e.type === "encounter";
}

export function isExtractedItem(e: ExtractionEntity): e is ExtractedItem {
  return e.type === "item";
}

export function isExtractedFaction(e: ExtractionEntity): e is ExtractedFaction {
  return e.type === "faction";
}

// --- Helper ---

export function buildExtractionSummary(batch: ExtractionBatchResult): ExtractionSummary {
  const summary: ExtractionSummary = {
    npcCount: 0,
    locationCount: 0,
    sceneSeedCount: 0,
    codexEntryCount: 0,
    encounterCount: 0,
    itemCount: 0,
    factionCount: 0,
    lowConfidenceCount: 0,
    needsReviewCount: 0,
  };
  for (const e of batch.entities) {
    if (e.type === "npc") summary.npcCount++;
    else if (e.type === "location") summary.locationCount++;
    else if (e.type === "scene_seed") summary.sceneSeedCount++;
    else if (e.type === "codex_entry") summary.codexEntryCount++;
    else if (e.type === "encounter") summary.encounterCount++;
    else if (e.type === "item") summary.itemCount++;
    else if (e.type === "faction") summary.factionCount++;
    if (e.confidence === "low") summary.lowConfidenceCount++;
    if (e.reviewStatus === "needs_review" || e.reviewStatus === "pending") summary.needsReviewCount++;
  }
  return summary;
}
