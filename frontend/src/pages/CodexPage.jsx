import React, { useState, useEffect } from "react";
import { useAppState } from "../context/AppStateContext";
import CodexScreen from "../components/codex/CodexScreen";
import { getCampaigns } from "../lib/api/codex";

export default function CodexPage({ campaignData: campaignDataProp, authFetch: authFetchProp }) {
  const appState = useAppState();
  const campaignData = campaignDataProp ?? appState.campaignData;
  const authFetch = authFetchProp ?? appState.authFetch;
  const setCampaignData = appState.setCampaignData;

  const [campaigns, setCampaigns] = useState([]);

  useEffect(() => {
    let cancelled = false;
    getCampaigns(authFetch)
      .then((list) => {
        if (!cancelled) setCampaigns(list);
      })
      .catch(() => {
        if (!cancelled) setCampaigns([]);
      });
    return () => { cancelled = true; };
  }, [authFetch]);

  const handleCampaignSelect = async (id) => {
    if (id == null) return;
    try {
      const res = await authFetch(`/api/campaigns/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      setCampaignData(data);
    } catch {
      // ignore
    }
  };

  return (
    <CodexScreen
      campaignData={campaignData}
      authFetch={authFetch}
      campaigns={campaigns}
      onCampaignSelect={handleCampaignSelect}
    />
  );
}
