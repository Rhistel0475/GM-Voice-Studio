/**
 * Shared campaign/session types for the Campaign Context layer.
 * Used by Live Board, Codex, NPC Workshop, and Voice Studio.
 * Backend: GET/PATCH /api/campaigns/:id for campaign load/save.
 */

/**
 * A single scene in a campaign (read-aloud, NPCs, items, reveals).
 * @typedef {Object} SceneTrigger
 * @property {string} name
 * @property {string} type
 * @property {string} [text]
 * @property {string|Object<string, unknown>} [action]
 * @property {string} [npc_name]
 * @property {string} [npc_id]
 * @property {string} [voice_id]
 */

/**
 * A single scene in a campaign (read-aloud, NPCs, items, reveals).
 * @typedef {Object} Scene
 * @property {string} [id]
 * @property {string} title
 * @property {string} [name]
 * @property {string} [description]
 * @property {string} [act]
 * @property {string} [type] - e.g. "combat" | "social" | "exploration" | "mystery" | "travel"
 * @property {string} [atmosphere_type] - e.g. "forest" | "tavern" | "town" | "dungeon" | "combat"
 * @property {string} [location]
 * @property {string} [read_aloud]
 * @property {string} [notes]
 * @property {string[]} [connected_scenes]
 * @property {string[]} [npcs] - NPC names present in this scene
 * @property {string[]} [items]
 * @property {string[]} [reveals]
 * @property {string[]} [codexRefs] - Optional codex entry IDs referenced by this scene. Backend: scene codex refs.
 * @property {SceneTrigger[]} [triggers]
 */

/**
 * Campaign data shape (aligns with current campaignData across the app).
 * Backend: GET /api/campaigns/:id, PATCH /api/campaigns/:id.
 * @typedef {Object} Campaign
 * @property {string|number} [id]
 * @property {string} title
 * @property {Scene[]} scenes
 * @property {Array<{ name: string, role?: string, personality?: string, voice_id?: string, [key: string]: unknown }>} npcs
 * @property {Array<{ name: string, hp?: string, ac?: string, [key: string]: unknown }>} [party]
 * @property {Array<{ name: string, [key: string]: unknown }>} [items]
 * @property {Array<{ name: string, type?: string, when?: string, [key: string]: unknown }>} [reveals]
 * @property {Array<string|{ name: string, description?: string }>} [locations]
 */

/**
 * One entry in the session action log (player, Co-DM, stat block, lore, system).
 * Backend: POST /api/session/events or websocket.
 * @typedef {Object} ActionLogEntry
 * @property {string} id
 * @property {string} role - e.g. "player" | "assistant" | "stat_block" | "lore" | "error"
 * @property {string} text
 * @property {string} [meta]
 */

/**
 * Active session state (scene index, session start).
 * Backend: session can be created/joined via API; scene index may be synced.
 * @typedef {Object} Session
 * @property {number|null} sessionStartMs
 * @property {number} activeSceneIndex
 */

/**
 * A narration clip for playback/list (TTS output).
 * Backend: store clips per session or campaign if needed.
 * @typedef {Object} NarrationClip
 * @property {string} id
 * @property {string} [voiceId]
 * @property {string} [text]
 * @property {string} [audioUrl]
 * @property {string} [createdAt]
 */

/**
 * Get NPCs that appear in a given scene (by matching scene.npcs names to campaign.npcs).
 * @param {Campaign|null} campaign
 * @param {Scene|null} scene
 * @returns {Array<{ name: string, role?: string, personality?: string, voice_id?: string }>}
 */
export function getNpcsForScene(campaign, scene) {
  if (!campaign?.npcs?.length || !scene?.npcs?.length) return [];
  const names = new Set(scene.npcs);
  return campaign.npcs.filter((n) => names.has(n.name));
}

/**
 * Get items that appear in a given scene (by matching scene.items to campaign.items).
 * @param {Campaign|null} campaign
 * @param {Scene|null} scene
 * @returns {Array<{ name: string }>}
 */
export function getItemsForScene(campaign, scene) {
  if (!campaign?.items?.length || !scene?.items?.length) return [];
  const names = new Set(scene.items);
  return campaign.items.filter((i) => names.has(i.name));
}

/**
 * Get reveals that appear in a given scene (by matching scene.reveals to campaign.reveals).
 * @param {Campaign|null} campaign
 * @param {Scene|null} scene
 * @returns {Array<{ name: string, type?: string }>}
 */
export function getRevealsForScene(campaign, scene) {
  if (!campaign?.reveals?.length || !scene?.reveals?.length) return [];
  const names = new Set(scene.reveals);
  return campaign.reveals.filter((r) => names.has(r.name));
}
