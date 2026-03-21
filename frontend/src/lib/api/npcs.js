/**
 * NPC API layer. Returns NPCProfile[] from campaign data or mock data.
 * When a dedicated backend /api/npcs exists, switch getNPCs to call it.
 */

/**
 * Map a minimal campaign NPC object to NPCProfile shape.
 * @param {Object} n - Raw npc from campaignData.npcs
 * @param {number} i - Index
 * @param {string} campaign - Campaign title
 * @returns {import("../../types/npc").NPCProfile}
 */
function campaignNpcToProfile(n, i, campaign) {
  const name = n.name || `NPC ${i + 1}`;
  const personality = n.personality || "";
  const traits = personality ? [personality.slice(0, 80) + (personality.length > 80 ? "…" : "")] : [];
  return {
    id: n.id || `npc-${i}-${String(name).replace(/\s+/g, "-")}`,
    name,
    role: n.role || "Unknown",
    profession: n.profession || n.role || "",
    faction: n.faction || "",
    location: n.location || "",
    personalityTraits: Array.isArray(n.personalityTraits) ? n.personalityTraits : traits,
    goals: Array.isArray(n.goals) ? n.goals : [],
    secrets: Array.isArray(n.secrets) ? n.secrets : [],
    quirks: Array.isArray(n.quirks) ? n.quirks : [],
    summary: n.summary || personality?.slice(0, 200) || n.role || "No description.",
    notes: n.notes || "",
    voiceId: n.voiceId || n.voice_id,
    portraitUrl: n.portraitUrl || n.portrait_url,
    disposition: n.disposition || "unknown",
    tags: Array.isArray(n.tags) ? n.tags : (n.role ? [n.role] : []),
    campaign: n.campaign || campaign,
    updatedAt: n.updatedAt || n.updated_at,
    favorite: !!n.favorite,
  };
}

/**
 * Get NPCs for the Workshop. Uses campaign npcs when present and non-empty; otherwise mock profiles.
 * @param {Object|null} campaignData - Legacy campaign payload (e.g. parsed JSON)
 * @param {Function} [authFetch] - Optional; for future GET /api/npcs
 * @returns {import("../../types/npc").NPCProfile[]}
 */
export function getNPCs(campaignData, _authFetch) {
  const npcs = campaignData?.npcs;
  if (npcs && Array.isArray(npcs) && npcs.length > 0) {
    const campaign = campaignData.title || "Campaign";
    return npcs.map((n, i) => campaignNpcToProfile(n, i, campaign));
  }
  return [];
}

/**
 * Save NPC (create or update). Stub: backend endpoint not yet implemented.
 * @param {import("../../types/npc").NPCProfile} _profile
 * @param {Function} _authFetch
 * @returns {Promise<null>}
 */
export async function saveNPC(_profile, _authFetch) {
  // TODO: implement when POST/PUT /api/npcs exists on the backend
  return null;
}

/**
 * Ask the backend for a provider-aware voice suggestion for an NPC already known to the campaign DB.
 * Returns null when the backend cannot suggest a voice yet.
 * @param {string} npcId
 * @param {Function} authFetch
 * @returns {Promise<Object|null>}
 */
export async function suggestNpcVoice(npcId, authFetch) {
  if (!npcId) return null;
  try {
    const res = await authFetch("/npc/suggest-voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ npc_id: npcId }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.suggested_voice ?? null;
  } catch {
    return null;
  }
}

/**
 * Push NPC to Live Board. Stub: backend endpoint not yet implemented.
 * @param {string} _profileId
 * @param {Function} _authFetch
 * @returns {Promise<false>}
 */
export async function pushToLiveBoard(_profileId, _authFetch) {
  // TODO: implement when POST /api/live-board/npc exists on the backend
  return false;
}
