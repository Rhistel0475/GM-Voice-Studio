import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import VoiceConsole from "../components/voices/VoiceConsole";
import NPCWorkshopPage from "./NPCWorkshopPage";

/**
 * VoicePage — container for voice management and NPC workshop.
 * Tabs: Voice Library (3-column console) | NPC Workshop
 */
export default function VoicePage({ authFetch: authFetchProp, campaignData: campaignDataProp }) {
  const [tab, setTab] = useState("voices");
  const appState = useAppState();
  const authFetch = authFetchProp ?? appState.authFetch;
  const campaignData = campaignDataProp ?? appState.campaignData;

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="tab-strip mb-3 flex-shrink-0">
        <button
          type="button"
          className={tab === "voices" ? "tab-active" : ""}
          onClick={() => setTab("voices")}
        >
          Voice Library
        </button>
        <button
          type="button"
          className={tab === "npcs" ? "tab-active" : ""}
          onClick={() => setTab("npcs")}
        >
          NPC Workshop
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-hidden">
        {tab === "voices" && (
          <VoiceConsole campaignData={campaignData} authFetch={authFetch} />
        )}
        {tab === "npcs" && (
          <NPCWorkshopPage campaignData={campaignData} authFetch={authFetch} />
        )}
      </div>
    </div>
  );
}
