import React, { useState, useMemo } from "react";
import CodexSidebar from "../components/codex/CodexSidebar";
import CodexDocumentViewer from "../components/codex/CodexDocumentViewer";
import CodexActionBar from "../components/codex/CodexActionBar";
import KnowledgeCard from "../components/codex/KnowledgeCard";

function documentContent(sidebarSection, campaignData, scenes, npcs) {
  if (sidebarSection === "Campaign") {
    return campaignData?.title
      ? { title: campaignData.title, body: `${scenes.length} scenes, ${npcs.length} NPCs. Use Adventures to open scene read-aloud.` }
      : { title: "No campaign", body: "Use Library to upload and parse adventure docs." };
  }
  if (sidebarSection === "Lore" || sidebarSection === "Rules") {
    return { title: sidebarSection, body: "RAG-driven content. Use Ask Question below with your campaign context." };
  }
  return null;
}

export default function CodexPage({ campaignData, authFetch }) {
  const [sidebarSection, setSidebarSection] = useState("Campaign");
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [askQuery, setAskQuery] = useState("");
  const [askResult, setAskResult] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [summarizeResult, setSummarizeResult] = useState("");
  const [isSummarizing, setIsSummarizing] = useState(false);

  const scenes = campaignData?.scenes?.length ? campaignData.scenes : [];
  const npcs = campaignData?.npcs?.length ? campaignData.npcs : [];
  const locations = campaignData?.locations?.length
    ? campaignData.locations
    : [...new Set(scenes.map((s) => s.location).filter(Boolean))];

  const doc = useMemo(
    () => documentContent(sidebarSection, campaignData, scenes, npcs),
    [sidebarSection, campaignData, scenes, npcs]
  );
  const displayDoc = selectedDoc || doc;

  const handleAskQuestion = async () => {
    if (!askQuery.trim() || isAsking) return;
    setIsAsking(true);
    setAskResult("");
    try {
      const res = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: askQuery.trim() }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setAskResult(data?.answer || data?.response || JSON.stringify(data));
    } catch (e) {
      setAskResult(e?.message || "Failed to get answer.");
    } finally {
      setIsAsking(false);
    }
  };

  const handleSummarize = async () => {
    if (!selectedDoc || isSummarizing) return;
    setIsSummarizing(true);
    setSummarizeResult("");
    try {
      const text =
        typeof selectedDoc === "string"
          ? selectedDoc
          : selectedDoc.read_aloud || selectedDoc.title || "";
      const res = await authFetch("/brain/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `Summarize the following for the GM:\n\n${text.slice(0, 4000)}`,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setSummarizeResult(data?.answer || data?.response || JSON.stringify(data));
    } catch (e) {
      setSummarizeResult(e?.message || "Summarize failed.");
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <section className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-3">
      <div className="xl:col-span-3 min-h-0 flex flex-col panel-ornate rounded border border-[#734f2c] p-2">
        <CodexSidebar
          section={sidebarSection}
          onSectionChange={setSidebarSection}
          scenes={scenes}
          locations={locations}
          npcs={npcs}
          selectedDoc={selectedDoc}
          onSelectDoc={setSelectedDoc}
        />
      </div>
      <div className="xl:col-span-9 min-h-0 flex flex-col gap-3 panel-ornate rounded border border-[#734f2c] p-3">
        <CodexDocumentViewer doc={displayDoc} />
        <CodexActionBar
          onSummarize={handleSummarize}
          summarizeDisabled={!selectedDoc}
          summarizeLoading={isSummarizing}
          askQuery={askQuery}
          onAskQueryChange={setAskQuery}
          onAskQuestion={handleAskQuestion}
          askDisabled={!askQuery.trim()}
          askLoading={isAsking}
        />
        {summarizeResult && (
          <KnowledgeCard title="Summary">{summarizeResult}</KnowledgeCard>
        )}
        {askResult && (
          <KnowledgeCard title="Answer" className="border-t border-[#a17a42] mt-2">
            {askResult}
          </KnowledgeCard>
        )}
      </div>
    </section>
  );
}
