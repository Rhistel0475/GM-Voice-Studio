import { useState } from "react";
import { useAppState } from "../context/AppStateContext";
import SettingsPage from "./SettingsPage";
import SectionHeader from "../components/layout/SectionHeader";
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
          <section className="max-w-xl mx-auto p-4 space-y-4">
            <SectionHeader title={campaignData?.title || "Campaign"} />

            {!hasCampaign ? (
              <ParchmentCard title="No Campaign Loaded">
                <p className="text-sm text-[var(--text-2)]">
                  Upload a campaign document in{" "}
                  <strong className="text-[var(--text-1)]">Prep → Upload Adventure</strong>{" "}
                  to get started. Parsed campaign data will appear here.
                </p>
              </ParchmentCard>
            ) : (
              <>
                <ParchmentCard title="Campaign Info">
                  <dl className="space-y-2">
                    {campaignData.system && (
                      <div className="flex justify-between text-sm">
                        <dt className="text-[var(--text-2)]">Game System</dt>
                        <dd className="text-[var(--text-1)] font-heading">{campaignData.system}</dd>
                      </div>
                    )}
                    {campaignData.setting && (
                      <div className="flex justify-between text-sm">
                        <dt className="text-[var(--text-2)]">Setting</dt>
                        <dd className="text-[var(--text-1)]">{campaignData.setting}</dd>
                      </div>
                    )}
                    {campaignData.description && (
                      <div className="pt-1">
                        <dt className="text-[10px] uppercase tracking-[0.15em] text-[var(--text-2)] mb-1">
                          Description
                        </dt>
                        <dd className="text-sm text-[var(--text-1)] leading-relaxed">
                          {campaignData.description}
                        </dd>
                      </div>
                    )}
                  </dl>
                </ParchmentCard>

                <ParchmentCard title="Content Summary">
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { label: "NPCs", count: npcCount },
                      { label: "Scenes", count: sceneCount },
                      { label: "Locations", count: locationCount },
                      { label: "Quests", count: questCount },
                      { label: "Factions", count: factionCount },
                    ].map(({ label, count }) => (
                      <div
                        key={label}
                        className="rounded-md border border-[#5c3e23] bg-[#1a1008]/80 px-3 py-2 text-center"
                      >
                        <div className="font-heading text-lg text-[var(--gold)]">{count}</div>
                        <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-2)]">
                          {label}
                        </div>
                      </div>
                    ))}
                  </div>
                </ParchmentCard>

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
