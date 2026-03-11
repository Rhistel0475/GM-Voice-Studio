import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import { useCampaignContextStore } from "../store/campaignContext";
import SectionHeader from "../components/layout/SectionHeader";
import { ParchmentCard } from "../components/shared";

export default function SettingsPage() {
  const { apiKey, setApiKey, requireApiKey, setCampaignData } = useAppState();
  const resetCampaignContext = useCampaignContextStore((s) => s.resetCampaignContext);
  const [cleared, setCleared] = useState(false);

  const handleClearCampaign = () => {
    // Clear localStorage FIRST so CampaignProvider effect does not re-import on reset
    ["gm_campaign_data", "gm_parse_result", "gm_parse_images", "gm_campaign_backend_id"].forEach(
      (k) => { try { localStorage.removeItem(k); } catch { /* ignore */ } }
    );
    // Then reset stores
    resetCampaignContext();
    setCampaignData(null);
    setCleared(true);
    setTimeout(() => setCleared(false), 3000);
  };

  return (
    <section className="max-w-xl mx-auto p-4 space-y-4">
      <SectionHeader title="Settings" />
      <ParchmentCard title="API Key">
        <p className="text-sm text-[var(--text-2)] mb-2">
          {requireApiKey
            ? "The server requires an API key. Enter it below to use Co-DM, voice, and AI features."
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
          Clear all loaded campaign data from memory and local storage. Use this to start fresh or load a new adventure.
        </p>
        <button
          type="button"
          className="cta-secondary"
          onClick={handleClearCampaign}
        >
          {cleared ? "Cleared!" : "Clear Campaign Data"}
        </button>
      </ParchmentCard>

      <p className="text-xs text-[var(--text-2)]">
        API key is stored in memory only. Campaign data is stored in localStorage.
      </p>
    </section>
  );
}
