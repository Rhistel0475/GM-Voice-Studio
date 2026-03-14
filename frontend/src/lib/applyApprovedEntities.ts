/**
 * Applies approved extraction review items to the campaign context store.
 *
 * Call this after the GM approves entities in the review queue. Each entity
 * type is mapped to the nearest campaign store model and upserted. Rejected
 * or un-reviewed items are ignored.
 */
import type {
  ExtractionReviewItem,
  ExtractedNPC,
  ExtractedLocation,
  ExtractedSceneSeed,
  ExtractedCodexEntry,
  ExtractedEncounter,
  ExtractedFaction,
  ExtractedItem,
} from "../types";
import type { Npc, Scene, CodexEntry, Campaign } from "../types";
import { createId } from "./utils/ids";

export interface ApplyEntityHandlers {
  upsertNpc: (npc: Npc) => void;
  upsertScene: (scene: Scene) => void;
  upsertCodexEntry: (entry: CodexEntry) => void;
  upsertCampaign: (campaign: Campaign) => void;
  setActiveCampaign: (id: string) => void;
  activeCampaignId: string | null;
  campaigns: Campaign[];
}

export interface ApplyResult {
  campaignId: string;
  npcCount: number;
  sceneCount: number;
  codexCount: number;
  totalApplied: number;
}

/**
 * Resolves or creates a campaign to attach entities to.
 * Uses the active campaign if one exists; otherwise creates a new one from
 * the document name and activates it.
 */
function resolveOrCreateCampaign(
  handlers: ApplyEntityHandlers,
  documentName: string
): string {
  if (handlers.activeCampaignId) return handlers.activeCampaignId;

  // Check if there's already exactly one campaign we can reuse.
  if (handlers.campaigns.length === 1) {
    handlers.setActiveCampaign(handlers.campaigns[0].id);
    return handlers.campaigns[0].id;
  }

  const campaignId = createId("campaign");
  const name = documentName || "Imported Campaign";
  handlers.upsertCampaign({ id: campaignId, name });
  handlers.setActiveCampaign(campaignId);
  return campaignId;
}

export function applyApprovedEntities(
  items: ExtractionReviewItem[],
  handlers: ApplyEntityHandlers,
  documentName = "Imported Campaign"
): ApplyResult {
  const approved = items.filter((i) => i.reviewStatus === "approved");
  if (approved.length === 0) {
    return { campaignId: handlers.activeCampaignId ?? "", npcCount: 0, sceneCount: 0, codexCount: 0, totalApplied: 0 };
  }

  const campaignId = resolveOrCreateCampaign(handlers, documentName);
  let npcCount = 0;
  let sceneCount = 0;
  let codexCount = 0;

  for (const item of approved) {
    const e = item.entity;

    if (e.type === "npc") {
      const npc = e as ExtractedNPC;
      handlers.upsertNpc({
        id: npc.id,
        campaignId,
        name: npc.name,
        summary: npc.summary,
        role: npc.role,
        profession: npc.profession,
        tags: npc.tags,
      });
      npcCount++;

    } else if (e.type === "location") {
      const loc = e as ExtractedLocation;
      const content = loc.notableFeatures.length
        ? loc.notableFeatures.join("\n")
        : undefined;
      handlers.upsertCodexEntry({
        id: loc.id,
        campaignId,
        type: "location",
        title: loc.name,
        summary: loc.summary,
        content,
        tags: loc.tags,
      });
      codexCount++;

    } else if (e.type === "scene_seed") {
      const seed = e as ExtractedSceneSeed;
      handlers.upsertScene({
        id: seed.id,
        campaignId,
        title: seed.title,
        summary: seed.summary || seed.setupText || "",
        npcIds: [],
        codexEntryIds: [],
        actionLogIds: [],
        narrationClipIds: [],
        tags: seed.tags,
      });
      sceneCount++;

    } else if (e.type === "codex_entry") {
      const entry = e as ExtractedCodexEntry;
      // Map extraction codex type → store codex type (best-effort).
      const storeType = (entry.entryType === "faction" ? "faction" : "lore") as CodexEntry["type"];
      handlers.upsertCodexEntry({
        id: entry.id,
        campaignId,
        type: storeType,
        title: entry.title,
        summary: entry.summary,
        content: entry.content,
        tags: entry.tags,
      });
      codexCount++;

    } else if (e.type === "encounter") {
      const enc = e as ExtractedEncounter;
      handlers.upsertCodexEntry({
        id: enc.id,
        campaignId,
        type: "lore",
        title: enc.name,
        summary: enc.summary || "",
        tags: enc.tags,
      });
      codexCount++;

    } else if (e.type === "faction") {
      const faction = e as ExtractedFaction;
      handlers.upsertCodexEntry({
        id: faction.id,
        campaignId,
        type: "faction",
        title: faction.name,
        summary: faction.description || faction.goals.join("; ") || "",
        tags: faction.tags,
      });
      codexCount++;

    } else if (e.type === "item") {
      const itm = e as ExtractedItem;
      handlers.upsertCodexEntry({
        id: itm.id,
        campaignId,
        type: "lore",
        title: itm.name,
        summary: itm.description || itm.narrativeImportance || "",
        tags: itm.tags,
      });
      codexCount++;
    }
  }

  return {
    campaignId,
    npcCount,
    sceneCount,
    codexCount,
    totalApplied: npcCount + sceneCount + codexCount,
  };
}
