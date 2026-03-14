/**
 * Campaign world builder: assemble extracted entities into a coherent campaign world structure.
 * Groups locations into regions, associates factions with regions, connects NPCs to locations/factions,
 * and generates a campaign overview. Prepares data for future UI visualization.
 */

import type {
  ExtractionEntity,
  ExtractedLocation,
  ExtractedFaction,
  ExtractedNPC,
  ExtractedSceneSeed,
  ExtractedEncounter,
  ExtractedItem,
  ExtractedCodexEntry,
} from "../types";
import {
  isExtractedLocation,
  isExtractedFaction,
  isExtractedNPC,
  isExtractedSceneSeed,
  isExtractedEncounter,
  isExtractedItem,
  isExtractedCodexEntry,
} from "../types";
import { resolveEntityRelationships, type ResolvedRelationship } from "./relationshipResolution";

export interface Region {
  id: string;
  name: string;
  locationIds: string[];
  factionIds: string[];
}

export interface CampaignWorldSummary {
  campaignId?: string;
  title: string;
  overview: string;
  regions: Region[];
  factionRegionMap: Record<string, string>; // factionId -> regionId
  npcLocationMap: Record<string, string[]>; // npcId -> locationIds
  npcFactionMap: Record<string, string[]>; // npcId -> factionIds
  relationships: ResolvedRelationship[];
  entityCounts: {
    locations: number;
    factions: number;
    npcs: number;
    scenes: number;
    encounters: number;
    items: number;
    codexEntries: number;
  };
}

function getEntityId(e: ExtractionEntity): string {
  return (e as { id?: string }).id ?? "";
}

function normalize(s: string): string {
  return (s || "").toLowerCase().trim();
}

/**
 * Heuristic: group locations into regions by name prefix or common token (e.g. "The Dark Woods" -> "Dark Woods").
 */
function groupLocationsIntoRegions(locations: ExtractedLocation[]): Region[] {
  const byRegion = new Map<string, string[]>();
  for (const loc of locations) {
    const name = loc.name.trim();
    const id = getEntityId(loc);
    // Use first significant token as region key (e.g. "Northern Vale" -> "Northern Vale")
    const tokens = name.split(/\s+/).filter(Boolean);
    const regionKey = tokens.length >= 2 ? tokens.slice(0, 2).join(" ") : name || "Unknown";
    const key = normalize(regionKey);
    if (!byRegion.has(key)) byRegion.set(key, []);
    byRegion.get(key)!.push(id);
  }
  return Array.from(byRegion.entries()).map(([key, locationIds], i) => ({
    id: `region-${i}-${key.replace(/\s+/g, "-")}`,
    name: key,
    locationIds,
    factionIds: [] as string[],
  }));
}

/**
 * Build a coherent campaign world from extracted entities.
 */
export function buildCampaignWorld(
  campaignEntities: ExtractionEntity[],
  options?: { campaignId?: string; title?: string }
): CampaignWorldSummary {
  const locations = campaignEntities.filter((e): e is ExtractedLocation => isExtractedLocation(e));
  const factions = campaignEntities.filter((e): e is ExtractedFaction => isExtractedFaction(e));
  const npcs = campaignEntities.filter((e): e is ExtractedNPC => isExtractedNPC(e));
  const scenes = campaignEntities.filter((e): e is ExtractedSceneSeed => isExtractedSceneSeed(e));
  const encounters = campaignEntities.filter((e): e is ExtractedEncounter => isExtractedEncounter(e));
  const items = campaignEntities.filter((e): e is ExtractedItem => isExtractedItem(e));
  const codexEntries = campaignEntities.filter((e): e is ExtractedCodexEntry => isExtractedCodexEntry(e));

  const relationships = resolveEntityRelationships(campaignEntities);

  const regions = groupLocationsIntoRegions(locations);
  const locationToRegion = new Map<string, string>();
  for (const r of regions) {
    for (const locId of r.locationIds) {
      locationToRegion.set(locId, r.id);
    }
  }

  const factionRegionMap: Record<string, string> = {};
  for (const rel of relationships) {
    if (rel.relation === "npc_to_location" && rel.toType === "location") {
      const regionId = locationToRegion.get(rel.toId);
      if (regionId) {
        // If an NPC is in a location, we could assign their faction to that region when we have faction->npc
      }
    }
  }
  for (const faction of factions) {
    const fid = getEntityId(faction);
    for (const ref of faction.locationRefs) {
      const locId = locations.find((l) => normalize(l.name) === normalize(ref.refName))?.id;
      if (locId) {
        const regionId = locationToRegion.get(locId);
        if (regionId) {
          factionRegionMap[fid] = regionId;
          const region = regions.find((r) => r.id === regionId);
          if (region && !region.factionIds.includes(fid)) {
            region.factionIds.push(fid);
          }
          break;
        }
      }
    }
  }

  const npcLocationMap: Record<string, string[]> = {};
  const npcFactionMap: Record<string, string[]> = {};
  for (const rel of relationships) {
    if (rel.fromType === "npc" && rel.relation === "npc_to_location") {
      const arr = npcLocationMap[rel.fromId] ?? [];
      if (!arr.includes(rel.toId)) arr.push(rel.toId);
      npcLocationMap[rel.fromId] = arr;
    }
    if (rel.fromType === "npc" && rel.relation === "npc_to_faction") {
      const arr = npcFactionMap[rel.fromId] ?? [];
      if (!arr.includes(rel.toId)) arr.push(rel.toId);
      npcFactionMap[rel.fromId] = arr;
    }
  }

  const title = options?.title ?? "Campaign World";
  const overview =
    `Campaign with ${locations.length} locations in ${regions.length} regions, ` +
    `${factions.length} factions, ${npcs.length} NPCs, ${scenes.length} scenes, ` +
    `${encounters.length} encounters, and ${items.length} items.`;

  return {
    campaignId: options?.campaignId,
    title,
    overview,
    regions,
    factionRegionMap,
    npcLocationMap,
    npcFactionMap,
    relationships,
    entityCounts: {
      locations: locations.length,
      factions: factions.length,
      npcs: npcs.length,
      scenes: scenes.length,
      encounters: encounters.length,
      items: items.length,
      codexEntries: codexEntries.length,
    },
  };
}
