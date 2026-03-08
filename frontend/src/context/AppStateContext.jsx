import React, { createContext, useCallback, useContext, useEffect, useState } from "react";

const AppStateContext = createContext(null);

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
  const [bannerState, setBannerState] = useState({
    sessionTime: "0:00",
    activeScene: "—",
    audioStatus: "idle",
  });

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

  const value = {
    campaignData,
    setCampaignData,
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
