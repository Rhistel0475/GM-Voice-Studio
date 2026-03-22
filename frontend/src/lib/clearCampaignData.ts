/**
 * Wipe campaign/adventure data app-wide: localStorage caches, zustand store,
 * extraction review queue, optional backend campaign DELETEs.
 */
import { setBackendCampaignId } from "./campaignPersistence";
import { useCampaignContextStore } from "../store/campaignContext";
import { useExtractionReviewQueueStore } from "../store/extractionReview";

export const CAMPAIGN_STORAGE_KEYS = ["gm_campaign_data", "gm_parse_result", "gm_parse_images"] as const;

export interface ClearCampaignDataOptions {
  /** When true, DELETE each campaign returned by GET /api/campaigns */
  deleteBackendCampaigns?: boolean;
  /** Prefer this key for API calls (e.g. Library page local key) before store.apiKey */
  xApiKey?: string;
}

export async function clearCampaignData(options: ClearCampaignDataOptions = {}): Promise<void> {
  const { deleteBackendCampaigns = false, xApiKey } = options;
  const store = useCampaignContextStore.getState();
  const apiKey = (xApiKey ?? store.apiKey ?? "").trim();

  const authFetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
    const headers = new Headers(init.headers || {});
    const key = apiKey;
    if (key) headers.set("X-API-Key", key);
    return fetch(input, { ...init, headers });
  };

  if (deleteBackendCampaigns) {
    try {
      const listResponse = await authFetch("/api/campaigns");
      const campaigns = listResponse.ok ? await listResponse.json() : [];
      if (Array.isArray(campaigns)) {
        for (const campaign of campaigns) {
          if (campaign?.id == null || campaign.id === "") continue;
          await authFetch(`/api/campaigns/${campaign.id}`, { method: "DELETE" });
        }
      }
    } catch {
      /* ignore backend cleanup errors during local reset */
    }
  }

  for (const key of CAMPAIGN_STORAGE_KEYS) {
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  }
  setBackendCampaignId(null);

  store.resetCampaignContext();
  useExtractionReviewQueueStore.getState().clearQueue();
}
