/**
 * Relationship resolution across extracted entities.
 * Resolves NPC↔location, NPC↔faction, scene↔encounter, encounter↔items, codex↔entities
 * using name matching, reference matching, and textual proximity.
 */

import type {
  ExtractionEntity,
  ExtractionRelationRef,
  ExtractedNPC,
  ExtractedLocation,
  ExtractedSceneSeed,
  ExtractedCodexEntry,
  ExtractedEncounter,
  ExtractedItem,
  ExtractedFaction,
} from "../types";
import {
  isExtractedNPC,
  isExtractedLocation,
  isExtractedSceneSeed,
  isExtractedCodexEntry,
  isExtractedEncounter,
  isExtractedItem,
  isExtractedFaction,
} from "../types";

export type ResolvedRelationType =
  | "npc_to_location"
  | "npc_to_faction"
  | "scene_to_encounter"
  | "encounter_to_item"
  | "codex_to_entity"
  | "location_to_region";

export interface ResolvedRelationship {
  fromType: string;
  fromId: string;
  toType: string;
  toId: string;
  relation: ResolvedRelationType | string;
  source: "name_match" | "ref_match" | "proximity";
}

function normalizeForMatch(s: string): string {
  return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function refNames(refs: ExtractionRelationRef[]): Set<string> {
  const set = new Set<string>();
  for (const r of refs) {
    if (r.refName) set.add(normalizeForMatch(r.refName));
  }
  return set;
}

function getEntityName(e: ExtractionEntity): string {
  switch (e.type) {
    case "npc":
      return (e as ExtractedNPC).name;
    case "location":
      return (e as ExtractedLocation).name;
    case "scene_seed":
      return (e as ExtractedSceneSeed).title;
    case "codex_entry":
      return (e as ExtractedCodexEntry).title;
    case "encounter":
      return (e as ExtractedEncounter).name;
    case "item":
      return (e as ExtractedItem).name;
    case "faction":
      return (e as ExtractedFaction).name;
    default:
      return "";
  }
}

function getEntityId(e: ExtractionEntity): string {
  return (e as { id?: string }).id ?? getEntityName(e);
}

/**
 * Resolve relationships across a mixed list of extracted entities.
 * Returns resolved links for NPCs↔locations, NPCs↔factions, scenes↔encounters,
 * encounters↔items, and codex↔referenced entities.
 */
export function resolveEntityRelationships(entities: ExtractionEntity[]): ResolvedRelationship[] {
  const relationships: ResolvedRelationship[] = [];
  const byType = {
    npc: entities.filter((e): e is ExtractedNPC => isExtractedNPC(e)),
    location: entities.filter((e): e is ExtractedLocation => isExtractedLocation(e)),
    scene_seed: entities.filter((e): e is ExtractedSceneSeed => isExtractedSceneSeed(e)),
    codex_entry: entities.filter((e): e is ExtractedCodexEntry => isExtractedCodexEntry(e)),
    encounter: entities.filter((e): e is ExtractedEncounter => isExtractedEncounter(e)),
    item: entities.filter((e): e is ExtractedItem => isExtractedItem(e)),
    faction: entities.filter((e): e is ExtractedFaction => isExtractedFaction(e)),
  };

  const locationNames = new Map<string, string>();
  byType.location.forEach((loc) => {
    locationNames.set(normalizeForMatch(loc.name), getEntityId(loc));
  });
  const factionNames = new Map<string, string>();
  byType.faction.forEach((f) => {
    factionNames.set(normalizeForMatch(f.name), getEntityId(f));
  });
  const sceneTitles = new Map<string, string>();
  byType.scene_seed.forEach((s) => {
    sceneTitles.set(normalizeForMatch(s.title), getEntityId(s));
  });
  const encounterNames = new Map<string, string>();
  byType.encounter.forEach((e) => {
    encounterNames.set(normalizeForMatch(e.name), getEntityId(e));
  });
  const itemNames = new Map<string, string>();
  byType.item.forEach((i) => {
    itemNames.set(normalizeForMatch(i.name), getEntityId(i));
  });
  const npcNames = new Map<string, string>();
  byType.npc.forEach((n) => {
    npcNames.set(normalizeForMatch(n.name), getEntityId(n));
  });

  // NPCs to locations (from locationRefs)
  for (const npc of byType.npc) {
    const npcId = getEntityId(npc);
    const locRefs = refNames(npc.locationRefs);
    for (const [norm, locId] of locationNames) {
      if (locRefs.has(norm)) {
        relationships.push({
          fromType: "npc",
          fromId: npcId,
          toType: "location",
          toId: locId,
          relation: "npc_to_location",
          source: "ref_match",
        });
      }
    }
  }

  // NPCs to factions
  for (const npc of byType.npc) {
    const npcId = getEntityId(npc);
    const factionRefs = refNames(npc.factionRefs);
    for (const [norm, factionId] of factionNames) {
      if (factionRefs.has(norm)) {
        relationships.push({
          fromType: "npc",
          fromId: npcId,
          toType: "faction",
          toId: factionId,
          relation: "npc_to_faction",
          source: "ref_match",
        });
      }
    }
  }

  // Scenes to encounters (by name/title overlap)
  for (const scene of byType.scene_seed) {
    const sceneId = getEntityId(scene);
    const sceneNorm = normalizeForMatch(scene.title);
    for (const [encNorm, encId] of encounterNames) {
      if (sceneNorm.includes(encNorm) || encNorm.includes(sceneNorm)) {
        relationships.push({
          fromType: "scene_seed",
          fromId: sceneId,
          toType: "encounter",
          toId: encId,
          relation: "scene_to_encounter",
          source: "name_match",
        });
      }
    }
  }

  // Encounters to items (treasureOrRewards vs item names)
  for (const enc of byType.encounter) {
    const encId = getEntityId(enc);
    const rewardSet = new Set(
      enc.treasureOrRewards.map((t) => normalizeForMatch(t))
    );
    for (const [itemNorm, itemId] of itemNames) {
      if (rewardSet.has(itemNorm) || [...rewardSet].some((r) => r.includes(itemNorm) || itemNorm.includes(r))) {
        relationships.push({
          fromType: "encounter",
          fromId: encId,
          toType: "item",
          toId: itemId,
          relation: "encounter_to_item",
          source: "ref_match",
        });
      }
    }
  }

  // Codex entries to NPCs, locations, factions
  for (const codex of byType.codex_entry) {
    const codexId = getEntityId(codex);
    const allRefs = [
      ...codex.npcRefs,
      ...codex.locationRefs,
      ...codex.sceneRefs,
      ...codex.factionRefs,
    ];
    const refSet = refNames(allRefs);
    for (const [norm, id] of npcNames) {
      if (refSet.has(norm)) {
        relationships.push({
          fromType: "codex_entry",
          fromId: codexId,
          toType: "npc",
          toId: id,
          relation: "codex_to_entity",
          source: "ref_match",
        });
      }
    }
    for (const [norm, id] of locationNames) {
      if (refSet.has(norm)) {
        relationships.push({
          fromType: "codex_entry",
          fromId: codexId,
          toType: "location",
          toId: id,
          relation: "codex_to_entity",
          source: "ref_match",
        });
      }
    }
    for (const [norm, id] of factionNames) {
      if (refSet.has(norm)) {
        relationships.push({
          fromType: "codex_entry",
          fromId: codexId,
          toType: "faction",
          toId: id,
          relation: "codex_to_entity",
          source: "ref_match",
        });
      }
    }
  }

  return relationships;
}
