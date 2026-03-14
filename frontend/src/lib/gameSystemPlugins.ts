/**
 * Game system plugin framework — support multiple tabletop systems via plugins.
 * Each plugin defines core rules, stat blocks, dice mechanics, encounter calculations.
 */

export interface StatBlockTemplate {
  id: string;
  name: string;
  fields: { key: string; label: string; type: "number" | "string" }[];
}

export interface DiceMechanic {
  id: string;
  name: string;
  notation: string; // e.g. "2d6+3"
  description?: string;
}

export interface EncounterCalculationRule {
  id: string;
  name: string;
  description: string;
  /** Formula or logic reference for difficulty/threat */
  formula?: string;
}

export interface GameSystemPlugin {
  systemId: string;
  name: string;
  version: string;
  coreRules: string[];
  statBlocks: StatBlockTemplate[];
  diceMechanics: DiceMechanic[];
  encounterCalculations: EncounterCalculationRule[];
}

const registry = new Map<string, GameSystemPlugin>();

/**
 * Register a game system plugin. Overwrites if systemId already exists.
 */
export function registerGameSystemPlugin(plugin: GameSystemPlugin): void {
  registry.set(plugin.systemId, plugin);
}

/**
 * Load a game system plugin by id. Returns undefined if not found.
 */
export function loadGameSystemPlugin(systemId: string): GameSystemPlugin | undefined {
  return registry.get(systemId);
}

/**
 * List all registered system ids.
 */
export function listRegisteredGameSystems(): string[] {
  return Array.from(registry.keys());
}
