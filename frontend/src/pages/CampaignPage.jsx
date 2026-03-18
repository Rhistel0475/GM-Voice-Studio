import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import SettingsPage from "./SettingsPage";
import { ParchmentCard } from "../components/shared";

/**
 * CampaignPage — campaign metadata and app settings.
 * Tabs: Campaign (overview + info) | Settings
 */
export default function CampaignPage() {
  const [tab, setTab] = useState("overview");
  const { campaignData } = useAppState();

  const npcCount = campaignData?.npcs?.length ?? 0;
  const sceneCount = campaignData?.scenes?.length ?? 0;
  const locationCount = campaignData?.locations?.length ?? 0;
  const questCount = campaignData?.quests?.length ?? 0;
  const factionCount = campaignData?.factions?.length ?? 0;
  const hasCampaign = Boolean(campaignData?.title);

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="tab-strip mb-3 flex-shrink-0">
        <button
          type="button"
          className={tab === "overview" ? "tab-active" : ""}
          onClick={() => setTab("overview")}
        >
          Campaign
        </button>
        <button
          type="button"
          className={tab === "settings" ? "tab-active" : ""}
          onClick={() => setTab("settings")}
        >
          Settings
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === "overview" && (
          <section className="max-w-3xl mx-auto p-5 space-y-5">

            {!hasCampaign ? (
              <div className="campaign-hero flex flex-col items-center text-center gap-4">
                <div className="font-heading text-[var(--gold)] text-xl tracking-widest uppercase">
                  No Campaign Loaded
                </div>
                <p className="text-sm text-[var(--text-2)] leading-relaxed max-w-sm">
                  Upload a campaign document in{" "}
                  <strong className="text-[var(--text-1)]">Prep → Upload Adventure</strong>{" "}
                  to get started. Parsed campaign data will appear here.
                </p>
              </div>
            ) : (
              <>
                {/* ── Campaign hero banner ── */}
                <div className="campaign-hero">
                  <div className="font-heading text-2xl text-[var(--gold)] tracking-widest uppercase leading-tight">
                    {campaignData.title}
                  </div>
                  {(campaignData.system || campaignData.setting) && (
                    <div className="flex gap-4 mt-2 flex-wrap">
                      {campaignData.system && (
                        <span className="text-xs text-[var(--text-2)] uppercase tracking-[0.14em]">
                          <span className="text-[var(--text-2)]">System: </span>
                          <span className="text-[var(--text-1)] font-heading">{campaignData.system}</span>
                        </span>
                      )}
                      {campaignData.setting && (
                        <span className="text-xs text-[var(--text-2)]">
                          <span className="text-[var(--text-2)]">Setting: </span>
                          <span className="text-[var(--text-1)]">{campaignData.setting}</span>
                        </span>
                      )}
                    </div>
                  )}
                  {campaignData.description && (
                    <p className="mt-3 text-sm text-[var(--text-1)] leading-relaxed border-t border-[#5c3e23]/50 pt-3">
                      {campaignData.description}
                    </p>
                  )}
                </div>

                {/* ── Stat tiles ── */}
                <div>
                  <div className="sidebar-section-label mb-3">Campaign Contents</div>
                  <div className="grid grid-cols-5 gap-2">
                    {[
                      { label: "NPCs", count: npcCount },
                      { label: "Scenes", count: sceneCount },
                      { label: "Locations", count: locationCount },
                      { label: "Quests", count: questCount },
                      { label: "Factions", count: factionCount },
                    ].map(({ label, count }) => (
                      <div key={label} className="stat-tile">
                        <div className="stat-tile-number">{count}</div>
                        <div className="stat-tile-label">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {campaignData.narrator_voice && (
                  <ParchmentCard title="Narrator Voice">
                    <p className="text-sm text-[var(--text-1)]">{campaignData.narrator_voice}</p>
                  </ParchmentCard>
                )}
              </>
            )}
          </section>
        )}
        {tab === "settings" && <SettingsPage />}
      </div>
    </div>
  );
}
