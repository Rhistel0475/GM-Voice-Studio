import React, { useState, useMemo } from "react";
import SectionHeader from "../layout/SectionHeader";
import CodexTabs from "./CodexTabs";
import CodexQuickView from "./CodexQuickView";

export default function CodexSidePanel({ campaignData, onInsertIntoNarration }) {
  const [codexTab, setCodexTab] = useState("documents");
  const [codexSelection, setCodexSelection] = useState(null);

  const scenes = campaignData?.scenes?.length ? campaignData.scenes : [];
  const npcs = campaignData?.npcs?.length ? campaignData.npcs : [];
  const locationsRaw = campaignData?.locations?.length ? campaignData.locations : [];
  const locations = locationsRaw.length ? locationsRaw : [...new Set(scenes.map((s) => s.location).filter(Boolean))];
  const documents = useMemo(
    () => [{ id: "campaign", title: campaignData?.title || "No campaign", summary: `${scenes.length} scenes` }],
    [campaignData?.title, scenes.length]
  );

  const handleTabChange = (tab) => {
    setCodexTab(tab);
    setCodexSelection(null);
  };

  return (
    <div className="h-full min-h-0 flex flex-col">
      <SectionHeader title="Codex" />
      <section className="panel-ornate flex-1 min-h-0 flex flex-col mt-2">
        <div className="panel-head">
          <div className="plaque">Codex</div>
        </div>
        <div className="panel-body min-h-0 flex flex-col">
          <CodexTabs selectedKey={codexTab} onChange={handleTabChange} />
          <CodexQuickView
            codexTab={codexTab}
            documents={documents}
            npcs={npcs}
            locations={locations}
            codexSelection={codexSelection}
            onSelect={setCodexSelection}
            onInsertIntoNarration={onInsertIntoNarration}
          />
        </div>
      </section>
      {codexTab === "documents" && scenes.length > 0 && (
        <div className="text-xs text-[var(--text-2)] mt-1">
          Scenes: {scenes.map((s) => s.title).filter(Boolean).join(", ") || "—"}
        </div>
      )}
    </div>
  );
}
