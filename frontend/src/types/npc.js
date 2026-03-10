/**
 * NPC types for the NPC Workshop. JSDoc for IDE support; repo uses .js.
 */

/**
 * @typedef {"friendly" | "neutral" | "hostile" | "unknown"} NPCDisposition
 */

/**
 * Full NPC profile (saved or from API).
 * @typedef {Object} NPCProfile
 * @property {string} id
 * @property {string} name
 * @property {string} role
 * @property {string} [profession]
 * @property {string} [faction]
 * @property {string} [location]
 * @property {string[]} personalityTraits
 * @property {string[]} goals
 * @property {string[]} secrets
 * @property {string[]} quirks
 * @property {string} summary
 * @property {string} [notes]
 * @property {string} [voiceId]
 * @property {string} [portraitUrl]
 * @property {NPCDisposition} [disposition]
 * @property {string[]} tags
 * @property {string} [campaign]
 * @property {string} [updatedAt]
 * @property {boolean} [favorite]
 */

/**
 * Draft state for the creation/editor form.
 * @typedef {Object} NPCDraft
 * @property {string} role
 * @property {string} profession
 * @property {string} faction
 * @property {string} location
 * @property {string[]} personalityTraits
 * @property {string[]} goals
 * @property {string[]} secrets
 * @property {string[]} quirks
 * @property {string} notes
 * @property {string} [preferredVoice]
 */

/**
 * Filter state for the roster sidebar.
 * @typedef {Object} NPCFilterState
 * @property {string} query
 * @property {string} [faction]
 * @property {string} [location]
 * @property {boolean} [favoritesOnly]
 */

/**
 * Default filter state for roster.
 * @returns {NPCFilterState}
 */
export function defaultNpcFilterState() {
  return {
    query: "",
    faction: "",
    location: "",
    favoritesOnly: false,
  };
}

/**
 * Default empty draft for new NPC.
 * @returns {NPCDraft}
 */
export function defaultNpcDraft() {
  return {
    role: "",
    profession: "",
    faction: "",
    location: "",
    personalityTraits: [],
    goals: [],
    secrets: [],
    quirks: [],
    notes: "",
    preferredVoice: "",
  };
}
