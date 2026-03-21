import { useState } from "react";
import { setBackendCampaignId } from "../lib/campaignPersistence";
import SectionHeader from "../components/layout/SectionHeader";
import { ParchmentCard } from "../components/shared";
import { useCampaignContextStore } from "../store/campaignContext";
import { useExtractionReviewQueueStore } from "../store/extractionReview";

const CAMPAIGN_STORAGE_KEYS = [
  "gm_campaign_data",
  "gm_parse_result",
  "gm_parse_images",
];

async function clearCampaignData({ deleteBackendCampaigns = false } = {}) {
  const store = useCampaignContextStore.getState();
  const apiKey = store.apiKey;

  const authFetch = (input, init = {}) => {
    const headers = new Headers(init.headers || {});
    const key = (apiKey || "").trim();
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

  CAMPAIGN_STORAGE_KEYS.forEach((key) => {
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  });
  setBackendCampaignId(null);

  store.resetCampaignContext();
  useExtractionReviewQueueStore.getState().clearQueue();
}

export default function SettingsPage() {
  const apiKey = useCampaignContextStore((s) => s.apiKey);
  const setApiKey = useCampaignContextStore((s) => s.setApiKey);
  const requireApiKey = useCampaignContextStore((s) => s.requireApiKey);
  const [cleared, setCleared] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  const handleClearCampaign = async () => {
    if (isClearing) return;
    if (!window.confirm("Clear all campaign data? This will remove your current parse, any loaded campaign, and delete all saved campaigns from the server. This cannot be undone.")) return;
    setIsClearing(true);
    try {
      await clearCampaignData({ deleteBackendCampaigns: true });
      setCleared(true);
      setTimeout(() => setCleared(false), 3000);
    } finally {
      setIsClearing(false);
    }
  };

  return (
    <section className="max-w-xl mx-auto p-4 space-y-4">
      <SectionHeader title="Settings" />
      <ParchmentCard title="API Key">
        <p className="text-sm text-[var(--text-2)] mb-2">
          {requireApiKey
            ? "The server requires an API key. Enter it below to use Co-GM, voice, and AI features."
            : "Optional. If the server is configured to require an API key, enter it here."}
        </p>
        <label className="field-wrap block">
          <span>API Key</span>
          <input
            type="password"
            className="chat-input w-full"
            placeholder="Enter API key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </label>
      </ParchmentCard>

      <ParchmentCard title="Campaign Data">
        <p className="text-sm text-[var(--text-2)] mb-3">
          Clear the active parse, shared campaign state, local storage caches, and saved backend campaigns so every page returns to a fresh state.
        </p>
        <button
          type="button"
          className="cta-secondary"
          onClick={handleClearCampaign}
          disabled={isClearing}
        >
          {isClearing ? "Clearing..." : cleared ? "Cleared!" : "Clear Campaign Data"}
        </button>
      </ParchmentCard>

      <p className="text-xs text-[var(--text-2)]">
        API key is stored in memory only. Campaign data is stored in localStorage.
      </p>
    </section>
  );
}
