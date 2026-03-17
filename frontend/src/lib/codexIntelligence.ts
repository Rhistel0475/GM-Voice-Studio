export type CodexIntelligenceTab = "npcs" | "scenes" | "locations" | "quests" | "factions";

export interface CodexIntelligenceEntry {
  id: string;
  tab: CodexIntelligenceTab;
  title: string;
  subtitle?: string;
  description: string;
  relationships: string[];
  speakTargetName?: string;
  narrationText: string;
  raw: Record<string, any>;
}

function toText(value: unknown): string {
  return String(value || "").trim();
}

function unique(values: Array<string | null | undefined>): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const text = toText(value);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) continue;
    seen.add(key);
    out.push(text);
  }
  return out;
}

function entityTitle(item: Record<string, any>): string {
  return toText(item.name || item.title || item.id);
}

function relatedViaGraph(
  campaignData: Record<string, any>,
  kind: "npc" | "scene" | "location" | "quest" | "faction",
  idOrName: string
): string[] {
  const relationships = Array.isArray(campaignData?.relationships) ? campaignData.relationships : [];
  const matches: string[] = [];
  for (const relationship of relationships) {
    if (!relationship || typeof relationship !== "object") continue;
    const fromType = toText(relationship.from_type);
    const toType = toText(relationship.to_type);
    const fromId = toText(relationship.from_id);
    const toId = toText(relationship.to_id);
    if (fromType === kind && fromId.toLowerCase() === idOrName.toLowerCase()) {
      matches.push(`${toType}: ${toId}`);
    } else if (toType === kind && toId.toLowerCase() === idOrName.toLowerCase()) {
      matches.push(`${fromType}: ${fromId}`);
    }
  }
  return unique(matches);
}

function buildNpcEntries(campaignData: Record<string, any>): CodexIntelligenceEntry[] {
  const npcs = Array.isArray(campaignData?.npcs) ? campaignData.npcs : [];
  return npcs
    .filter((npc) => npc && typeof npc === "object" && toText(npc.name))
    .map((npc) => {
      const title = toText(npc.name);
      const description = toText(npc.description || npc.personality || npc.summary);
      const relationships = unique([
        npc.role ? `role: ${npc.role}` : null,
        npc.faction ? `faction: ${npc.faction}` : null,
        ...(Array.isArray(npc.related_scenes) ? npc.related_scenes.map((value: unknown) => `scene: ${toText(value)}`) : []),
        ...(Array.isArray(npc.related_locations) ? npc.related_locations.map((value: unknown) => `location: ${toText(value)}`) : []),
        ...(Array.isArray(npc.related_quests) ? npc.related_quests.map((value: unknown) => `quest: ${toText(value)}`) : []),
        ...relatedViaGraph(campaignData, "npc", title),
      ]);
      return {
        id: toText(npc.id || title),
        tab: "npcs" as const,
        title,
        subtitle: toText(npc.role),
        description,
        relationships,
        speakTargetName: title,
        narrationText: description || title,
        raw: npc,
      };
    });
}

function buildSceneEntries(campaignData: Record<string, any>): CodexIntelligenceEntry[] {
  const scenes = Array.isArray(campaignData?.scenes) ? campaignData.scenes : [];
  return scenes
    .filter((scene) => scene && typeof scene === "object" && toText(scene.title || scene.name))
    .map((scene) => {
      const title = toText(scene.title || scene.name);
      const npcs = Array.isArray(scene.npcs) ? scene.npcs.map((value: unknown) => toText(value)) : [];
      const description = toText(scene.read_aloud || scene.notes || scene.summary || scene.description);
      const location = toText(scene.location);
      const speakTargetName = npcs.find(Boolean);
      return {
        id: toText(scene.id || title),
        tab: "scenes" as const,
        title,
        subtitle: location || undefined,
        description,
        relationships: unique([
          ...npcs.map((name) => `npc: ${name}`),
          location ? `location: ${location}` : null,
          ...(Array.isArray(scene.related_quests) ? scene.related_quests.map((value: unknown) => `quest: ${toText(value)}`) : []),
          ...(Array.isArray(scene.related_factions) ? scene.related_factions.map((value: unknown) => `faction: ${toText(value)}`) : []),
          ...relatedViaGraph(campaignData, "scene", title),
        ]),
        speakTargetName,
        narrationText: description || [title, location].filter(Boolean).join(" at "),
        raw: scene,
      };
    });
}

function buildLocationEntries(campaignData: Record<string, any>): CodexIntelligenceEntry[] {
  const locations = Array.isArray(campaignData?.locations) ? campaignData.locations : [];
  return locations
    .map((location) => (typeof location === "string" ? { name: location } : location))
    .filter((location) => location && typeof location === "object" && toText(location.name))
    .map((location) => {
      const title = toText(location.name);
      const description = toText(location.description || location.summary);
      const relatedNpcs = Array.isArray(location.related_npcs) ? location.related_npcs.map((value: unknown) => toText(value)) : [];
      return {
        id: toText(location.id || title),
        tab: "locations" as const,
        title,
        description,
        relationships: unique([
          ...relatedNpcs.map((name) => `npc: ${name}`),
          ...(Array.isArray(location.related_scenes) ? location.related_scenes.map((value: unknown) => `scene: ${toText(value)}`) : []),
          ...(Array.isArray(location.related_quests) ? location.related_quests.map((value: unknown) => `quest: ${toText(value)}`) : []),
          ...(Array.isArray(location.related_factions) ? location.related_factions.map((value: unknown) => `faction: ${toText(value)}`) : []),
          ...relatedViaGraph(campaignData, "location", title),
        ]),
        speakTargetName: relatedNpcs.find(Boolean),
        narrationText: description || title,
        raw: location,
      };
    });
}

function buildQuestEntries(campaignData: Record<string, any>): CodexIntelligenceEntry[] {
  const quests = Array.isArray(campaignData?.quests) ? campaignData.quests : [];
  return quests
    .filter((quest) => quest && typeof quest === "object" && entityTitle(quest))
    .map((quest) => {
      const title = entityTitle(quest);
      const description = toText(quest.description || quest.summary || quest.objective || quest.content);
      const relatedNpcs = Array.isArray(quest.related_npcs) ? quest.related_npcs.map((value: unknown) => toText(value)) : [];
      return {
        id: toText(quest.id || title),
        tab: "quests" as const,
        title,
        subtitle: toText(quest.stakes),
        description,
        relationships: unique([
          ...relatedNpcs.map((name) => `npc: ${name}`),
          ...(Array.isArray(quest.related_locations) ? quest.related_locations.map((value: unknown) => `location: ${toText(value)}`) : []),
          ...(Array.isArray(quest.related_scenes) ? quest.related_scenes.map((value: unknown) => `scene: ${toText(value)}`) : []),
          ...relatedViaGraph(campaignData, "quest", title),
        ]),
        speakTargetName: relatedNpcs.find(Boolean),
        narrationText: [description, toText(quest.objective), toText(quest.stakes)].filter(Boolean).join(" "),
        raw: quest,
      };
    });
}

function buildFactionEntries(campaignData: Record<string, any>): CodexIntelligenceEntry[] {
  const factions = Array.isArray(campaignData?.factions) ? campaignData.factions : [];
  return factions
    .filter((faction) => faction && typeof faction === "object" && entityTitle(faction))
    .map((faction) => {
      const title = entityTitle(faction);
      const description = toText(faction.description || faction.summary || faction.content);
      const relatedNpcs = Array.isArray(faction.related_npcs) ? faction.related_npcs.map((value: unknown) => toText(value)) : [];
      return {
        id: toText(faction.id || title),
        tab: "factions" as const,
        title,
        description,
        relationships: unique([
          ...relatedNpcs.map((name) => `npc: ${name}`),
          ...(Array.isArray(faction.related_locations) ? faction.related_locations.map((value: unknown) => `location: ${toText(value)}`) : []),
          ...(Array.isArray(faction.related_scenes) ? faction.related_scenes.map((value: unknown) => `scene: ${toText(value)}`) : []),
          ...relatedViaGraph(campaignData, "faction", title),
        ]),
        speakTargetName: relatedNpcs.find(Boolean),
        narrationText: [title, description].filter(Boolean).join(": "),
        raw: faction,
      };
    });
}

export function buildCodexIntelligence(campaignData: Record<string, any> | null | undefined) {
  const safeData = campaignData && typeof campaignData === "object" ? campaignData : {};
  return {
    npcs: buildNpcEntries(safeData),
    scenes: buildSceneEntries(safeData),
    locations: buildLocationEntries(safeData),
    quests: buildQuestEntries(safeData),
    factions: buildFactionEntries(safeData),
  };
}
