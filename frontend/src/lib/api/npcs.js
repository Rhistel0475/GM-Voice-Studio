/**
 * NPC API layer. Returns NPCProfile[] from campaign data or mock data.
 * When a dedicated backend /api/npcs exists, switch getNPCs to call it.
 */

import { MOCK_NPC_PROFILES } from "../utils/mockData";

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
 * @param {Object|null} campaignData - From AppStateContext
 * @param {Function} [authFetch] - Optional; for future GET /api/npcs
 * @returns {import("../../types/npc").NPCProfile[]}
 */
export function getNPCs(campaignData, authFetch) {
  const npcs = campaignData?.npcs;
  if (npcs && Array.isArray(npcs) && npcs.length > 0) {
    const campaign = campaignData.title || "Campaign";
    return npcs.map((n, i) => campaignNpcToProfile(n, i, campaign));
  }
  return MOCK_NPC_PROFILES;
}

/**
 * Save NPC (create or update). Stub: call backend when endpoint exists; otherwise no-op or local toast.
 * @param {import("../../types/npc").NPCProfile} profile
 * @param {Function} authFetch
 * @returns {Promise<import("../../types/npc").NPCProfile | null>}
 */
export async function saveNPC(profile, authFetch) {
  try {
    const res = await authFetch("/api/npcs", {
      method: profile.id && profile.id.startsWith("mock-") ? "POST" : "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });
    if (res.ok) {
      const data = await res.json();
      return data;
    }
  } catch {
    // Backend may not have endpoint yet
  }
  return null;
}

/**
 * Push NPC to Live Board. Stub: toast "Coming soon" until backend supports it.
 * @param {string} profileId
 * @param {Function} authFetch
 * @returns {Promise<boolean>}
 */
export async function pushToLiveBoard(profileId, authFetch) {
  try {
    const res = await authFetch("/api/live-board/npc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ npc_id: profileId }),
    });
    return res.ok;
  } catch {
    return false;
  }
}
