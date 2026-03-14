import { useEffect, useState } from "react";
import { useAppState } from "../context/AppStateContext";
import { useCampaignOptional } from "../context/CampaignContext";
import { useCodexEntriesForActiveCampaign } from "../store/selectors";
import CodexScreen from "../components/codex/CodexScreen";
import { getCampaigns } from "../lib/api/codex";

export default function CodexPage({ campaignData: campaignDataProp, authFetch: authFetchProp }) {
  const appState = useAppState();
  const campaignData = campaignDataProp ?? appState.campaignData;
  const authFetch = authFetchProp ?? appState.authFetch;
  const setCampaignData = appState.setCampaignData;
  const campaignCtx = useCampaignOptional();
  const codexEntriesFromStore = useCodexEntriesForActiveCampaign();

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
      onAddToLiveBoard={
        campaignCtx
          ? (entry) => campaignCtx.addCodexEntryToScene(entry?.id ?? entry?.title, campaignCtx.activeSceneIndex)
          : undefined
      }
      codexEntriesFromStore={campaignCtx ? codexEntriesFromStore : undefined}
    />
  );
}
