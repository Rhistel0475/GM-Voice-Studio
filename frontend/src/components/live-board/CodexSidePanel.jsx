import { useEffect, useMemo, useState } from "react";
import CodexTabs from "./CodexTabs";
import CodexQuickView from "./CodexQuickView";
import CampaignBrainPanel from "./CampaignBrainPanel";
import { buildCodexIntelligence } from "../../lib/codexIntelligence";

/**
 * Right panel: tabs (Documents, NPCs, Locations, Rules) and quick preview cards.
 * Section header "Codex" is rendered by LiveBoardPage.
 */
export default function CodexSidePanel({
  campaignData,
  onInsertIntoNarration,
  onSpeakNpc,
  authFetch,
  onNarrateAnswer,
}) {
  const [codexTab, setCodexTab] = useState("npcs");
  const [codexSelection, setCodexSelection] = useState(null);
  const [brainDocuments, setBrainDocuments] = useState(() => campaignData?.documents || []);

  useEffect(() => {
    setBrainDocuments(campaignData?.documents || []);
  }, [campaignData?.documents, campaignData?.id]);

  const intelligence = useMemo(() => buildCodexIntelligence(campaignData), [campaignData]);
  const entries = intelligence[codexTab] || [];

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
        <div className="panel-body min-h-0 flex flex-col gap-2 overflow-hidden">
          <CampaignBrainPanel
            campaignId={campaignData?.id}
            authFetch={authFetch}
            documents={brainDocuments}
            onDocumentsChange={setBrainDocuments}
            onNarrateAnswer={onNarrateAnswer}
          />
          <CodexTabs selectedKey={codexTab} onChange={handleTabChange} />
          <CodexQuickView
            codexTab={codexTab}
            entries={entries}
            codexSelection={codexSelection}
            onSelect={setCodexSelection}
            onInsertIntoNarration={onInsertIntoNarration}
            onSpeakNpc={onSpeakNpc}
          />
        </div>
      </section>
      {entries.length > 0 && (
        <p className="text-xs text-[var(--text-2)] mt-2 px-1">
          {entries.length} campaign knowledge entr{entries.length !== 1 ? "ies" : "y"} in this view.
        </p>
      )}
    </div>
  );
}
