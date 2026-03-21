import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { setBackendCampaignId } from "../lib/campaignPersistence";
import { useCampaignContextStore } from "../store/campaignContext";
import { useExtractionReviewQueueStore } from "../store/extractionReview";

const AppStateContext = createContext(null);
const DEFAULT_BANNER_STATE = {
  sessionTime: "0:00",
  activeScene: "—",
  audioStatus: "idle",
};
const CAMPAIGN_STORAGE_KEYS = [
  "gm_campaign_data",
  "gm_parse_result",
  "gm_parse_images",
];

export function AppStateProvider({ children }) {
  const [campaignData, setCampaignDataState] = useState(() => {
    try {
      const saved = localStorage.getItem("gm_campaign_data");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [requireApiKey, setRequireApiKey] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [bannerState, setBannerState] = useState(DEFAULT_BANNER_STATE);

  const setCampaignData = useCallback((data) => {
    setCampaignDataState(data);
    try {
      if (data != null) {
        localStorage.setItem("gm_campaign_data", JSON.stringify(data));
      } else {
        localStorage.removeItem("gm_campaign_data");
      }
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/config")
      .then((r) => (r.ok ? r.json() : { require_api_key: false }))
      .then((cfg) => {
        if (!cancelled) setRequireApiKey(Boolean(cfg?.require_api_key));
      })
      .catch(() => {
        if (!cancelled) setRequireApiKey(false);
      });
    return () => { cancelled = true; };
  }, []);

  const getAuthHeaders = useCallback(() => {
    const key = apiKey.trim();
    return key ? { "X-API-Key": key } : {};
  }, [apiKey]);

  const authFetch = useCallback(
    (input, init = {}) => {
      const headers = new Headers(init.headers || {});
      Object.entries(getAuthHeaders()).forEach(([k, v]) => headers.set(k, v));
      return fetch(input, { ...init, headers });
    },
    [getAuthHeaders]
  );

  const clearCampaignData = useCallback(async ({ deleteBackendCampaigns = false } = {}) => {
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

    CAMPAIGN_STORAGE_KEYS.forEach((key) => {
      try {
        localStorage.removeItem(key);
      } catch {
        /* ignore */
      }
    });
    setBackendCampaignId(null);

    useCampaignContextStore.getState().resetCampaignContext();
    useExtractionReviewQueueStore.getState().clearQueue();
    setCampaignDataState(null);
    setBannerState(DEFAULT_BANNER_STATE);
  }, [authFetch]);

  const value = {
    campaignData,
    setCampaignData,
    clearCampaignData,
    requireApiKey,
    apiKey,
    setApiKey,
    authFetch,
    bannerState,
    setBannerState,
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
