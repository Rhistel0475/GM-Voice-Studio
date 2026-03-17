import type { CampaignSystemPreset } from "../types";

export const DEFAULT_GAME_SYSTEM_ID = "homebrew";

const aliases: Record<string, string> = {
  "d&d": "dnd",
  "dnd5e": "dnd",
  "5e": "dnd",
  "call_of_cthulhu": "coc",
  "call-of-cthulhu": "coc",
  "callofcthulhu": "coc",
};

const builtInSystems: CampaignSystemPreset[] = [
  {
    id: "dnd",
    label: "D&D",
    rules_flavor: "Heroic fantasy, class-driven abilities, spellcasting, and adventure-forward escalation.",
    skill_check_terminology: {
      skill_term: "skill",
      check_term: "ability check",
      difficulty_term: "DC",
    },
    encounter_assumptions: "Expect set-piece combat, clear monsters, and high-action scene framing.",
    thematic_guidance: "Favor treasure, exploration, factions, quests, and big heroic turns.",
  },
  {
    id: "pathfinder",
    label: "Pathfinder",
    rules_flavor: "Detailed fantasy adventure with crunchy tactics, feats, and precise mechanical hooks.",
    skill_check_terminology: {
      skill_term: "skill",
      check_term: "check",
      difficulty_term: "DC",
    },
    encounter_assumptions: "Encounters often assume stronger tactical structure and richer build expression.",
    thematic_guidance: "Support layered prep, deeper world detail, and tactically expressive conflicts.",
  },
  {
    id: "coc",
    label: "Call of Cthulhu",
    rules_flavor: "Investigative horror driven by clues, dread, fragile investigators, and mounting consequences.",
    skill_check_terminology: {
      skill_term: "skill",
      check_term: "roll",
      difficulty_term: "difficulty level",
    },
    encounter_assumptions: "Combat is dangerous, investigation matters, and tension often outweighs open fights.",
    thematic_guidance: "Lean into unease, secrets, paranoia, and clue-rich scene progression.",
  },
  {
    id: "homebrew",
    label: "Homebrew",
    rules_flavor: "Custom world and table rules defined by the GM rather than a preset chassis.",
    skill_check_terminology: {
      skill_term: "skill or trait",
      check_term: "check or roll",
      difficulty_term: "difficulty",
    },
    encounter_assumptions: "Avoid assuming class structures, challenge math, or familiar stat formats unless the source says so.",
    thematic_guidance: "Follow the campaign's own terminology and tone instead of forcing a default fantasy template.",
  },
];

const registry = new Map<string, CampaignSystemPreset>(
  builtInSystems.map((system) => [system.id, system])
);

export function normalizeGameSystemId(systemId?: string | null): string {
  const raw = String(systemId || "").trim().toLowerCase().replace(/\s+/g, "_");
  if (!raw) return DEFAULT_GAME_SYSTEM_ID;
  if (registry.has(raw)) return raw;
  const alias = aliases[raw] || aliases[raw.replace(/_/g, "")];
  if (alias && registry.has(alias)) return alias;
  return DEFAULT_GAME_SYSTEM_ID;
}

export function registerGameSystemPlugin(plugin: CampaignSystemPreset): void {
  registry.set(normalizeGameSystemId(plugin.id), { ...plugin, id: normalizeGameSystemId(plugin.id) });
}

export function normalizeGameSystemPlugin(value: unknown): CampaignSystemPreset | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Partial<CampaignSystemPreset>;
  const id = normalizeGameSystemId(raw.id);
  const fallback = registry.get(id) || registry.get(DEFAULT_GAME_SYSTEM_ID);
  if (!fallback) return null;
  return {
    id,
    label: String(raw.label || fallback.label),
    rules_flavor: String(raw.rules_flavor || fallback.rules_flavor),
    skill_check_terminology: {
      skill_term: String(raw.skill_check_terminology?.skill_term || fallback.skill_check_terminology.skill_term),
      check_term: String(raw.skill_check_terminology?.check_term || fallback.skill_check_terminology.check_term),
      difficulty_term: String(raw.skill_check_terminology?.difficulty_term || fallback.skill_check_terminology.difficulty_term),
    },
    encounter_assumptions: String(raw.encounter_assumptions || fallback.encounter_assumptions),
    thematic_guidance: String(raw.thematic_guidance || fallback.thematic_guidance),
  };
}

export function listGameSystemPlugins(): CampaignSystemPreset[] {
  return Array.from(registry.values());
}

export function loadGameSystemPlugin(systemId?: string | null): CampaignSystemPreset | undefined {
  return registry.get(normalizeGameSystemId(systemId));
}

export function resolveGameSystemPlugin(
  systemId?: string | null,
  systems?: CampaignSystemPreset[]
): CampaignSystemPreset {
  const normalizedId = normalizeGameSystemId(systemId);
  const inList = Array.isArray(systems) ? systems.find((system) => system.id === normalizedId) : undefined;
  return inList || loadGameSystemPlugin(normalizedId) || loadGameSystemPlugin(DEFAULT_GAME_SYSTEM_ID) || builtInSystems[0];
}

export function listRegisteredGameSystems(): string[] {
  return listGameSystemPlugins().map((system) => system.id);
}
