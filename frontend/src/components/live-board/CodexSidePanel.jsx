import { useState, useMemo } from "react";
import CodexTabs from "./CodexTabs";
import CodexQuickView from "./CodexQuickView";

/**
 * Right panel: tabs (Documents, NPCs, Locations, Rules) and quick preview cards.
 * Section header "Codex" is rendered by LiveBoardPage.
 */
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
    <div className="h-full min-h-0 flex flex-col rounded-b-lg overflow-hidden">
      <section className="panel-ornate flex-1 min-h-0 flex flex-col">
        <div className="panel-head">
          <div className="plaque">Quick reference</div>
        </div>
        <div className="panel-body min-h-0 flex flex-col gap-2">
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
        <p className="text-xs text-[var(--text-2)] mt-2 px-1">
          Scenes: {scenes.map((s) => s.title).filter(Boolean).join(", ") || "—"}
        </p>
      )}
    </div>
  );
}
