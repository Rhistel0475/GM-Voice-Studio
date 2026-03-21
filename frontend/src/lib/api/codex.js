/**
 * Codex API layer. Returns CodexItem[] from campaign data or mock data.
 * When a dedicated backend codex endpoint exists, switch this to call it.
 */

import { MOCK_CODEX_ITEMS } from "../utils/mockData";

/**
 * Map campaignData (scenes, npcs, locations) to CodexItem shape.
 * @param {Object} campaignData - Legacy campaign payload (e.g. parsed JSON)
 * @returns {import("../../types/codex").CodexItem[]}
 */
function campaignToCodexItems(campaignData) {
  if (!campaignData) return [];
  const campaign = campaignData.title || "Campaign";
  const items = [];

  const scenes = campaignData.scenes || [];
  scenes.forEach((s, i) => {
    items.push({
      id: `scene-${i}-${s.title || i}`,
      title: s.title || `Scene ${i + 1}`,
      category: "document",
      tags: s.type ? [s.type] : [],
      summary: s.read_aloud ? s.read_aloud.slice(0, 120) + (s.read_aloud.length > 120 ? "…" : "") : `${s.notes || "Scene"} (${scenes.length} scenes)`,
      excerpt: s.read_aloud?.slice(0, 200),
      content: s.read_aloud || s.notes || "",
      campaign,
      source: "Adventures",
    });
  });

  const npcs = campaignData.npcs || [];
  npcs.forEach((n, i) => {
    const name = n.name || `NPC ${i + 1}`;
    items.push({
      id: `npc-${i}-${name}`,
      title: name,
      category: "npc",
      tags: n.role ? [n.role] : [],
      summary: n.personality?.slice(0, 120) || n.role || "No description.",
      excerpt: n.personality?.slice(0, 200),
      content: n.personality || "",
      campaign,
      source: "NPCs",
    });
  });

  const locations = campaignData.locations || [];
  locations.forEach((loc, i) => {
    const name = typeof loc === "string" ? loc : loc.name || `Location ${i + 1}`;
    items.push({
      id: `loc-${i}-${name}`,
      title: name,
      category: "location",
      tags: [],
      summary: typeof loc === "object" && loc.description ? loc.description.slice(0, 120) : `Location: ${name}`,
      content: typeof loc === "object" && loc.description ? loc.description : "",
      campaign,
      source: "Locations",
    });
  });

  return items;
}

/**
 * Get codex items for the research view. Uses campaign data when present and non-empty; otherwise mock data.
 * @param {Object|null} campaignData - From context
 * @param {Function} [authFetch] - Optional; for future backend codex API
 * @returns {import("../../types/codex").CodexItem[]}
 */
export function getCodexItems(campaignData, _authFetch) {
  const fromCampaign = campaignToCodexItems(campaignData);
  if (fromCampaign.length > 0) return fromCampaign;
  return MOCK_CODEX_ITEMS;
}

/**
 * Fetch campaigns list for the campaign selector. Use with authFetch from context.
 * @param {Function} authFetch
 * @returns {Promise<Array<{ id: number, title?: string }>>}
 */
export async function getCampaigns(authFetch) {
  const res = await authFetch("/api/campaigns");
  if (!res.ok) return [];
  const data = await res.json();
  return Array.isArray(data) ? data : [];
}
