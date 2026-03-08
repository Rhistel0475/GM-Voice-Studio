/**
 * @typedef {"document" | "npc" | "location" | "rule" | "lore" | "faction"} CodexCategory
 */

/**
 * @typedef {Object} CodexItem
 * @property {string} id
 * @property {string} title
 * @property {CodexCategory} category
 * @property {string[]} [tags]
 * @property {string} summary
 * @property {string} [excerpt]
 * @property {string} [content]
 * @property {string} [campaign]
 * @property {string} [updatedAt]
 * @property {string} [source]
 */

/**
 * @typedef {Object} CodexFilterState
 * @property {string} query
 * @property {CodexCategory | "all"} category
 * @property {string[]} tags
 */

/** @type {CodexCategory[]} */
export const CODEX_CATEGORIES = [
  "document",
  "npc",
  "location",
  "rule",
  "lore",
  "faction",
];

/**
 * Default filter state.
 * @returns {CodexFilterState}
 */
export function defaultCodexFilterState() {
  return {
    query: "",
    category: "all",
    tags: [],
  };
}
