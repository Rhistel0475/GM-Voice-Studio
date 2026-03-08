import React from "react";
import SectionHeader from "../components/layout/SectionHeader";
import GMControlPanel from "../components/live-board/GMControlPanel";
import SessionStream from "../components/live-board/SessionStream";
import CodexSidePanel from "../components/live-board/CodexSidePanel";

/**
 * Live Board three-column layout: GM Control | Live Session | Codex.
 * middleColumn: React node (e.g. <MiddleColumn ... /> from App.jsx) for the center column.
 */
export default function LiveBoardPage({
  campaignData,
  scene,
  selectedNpcName,
  onSelectNpc,
  onInsertIntoNarration,
  middleColumn,
}) {
  return (
    <section className="min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-3">
      <div className="xl:col-span-3 min-h-0 flex flex-col">
        <SectionHeader title="GM Control" className="text-center mb-1" />
        <div className="min-h-0 flex-1">
          <GMControlPanel
            campaignData={campaignData}
            scene={scene}
            selectedNpcName={selectedNpcName}
            onSelectNpc={onSelectNpc}
          />
        </div>
      </div>
      <div className="xl:col-span-5 min-h-0 flex flex-col">
        <div className="min-h-0 flex-1">
          <SessionStream>{middleColumn}</SessionStream>
        </div>
      </div>
      <div className="xl:col-span-4 min-h-0 flex flex-col">
        <SectionHeader title="Codex" className="text-center mb-1" />
        <div className="min-h-0 flex-1">
          <CodexSidePanel campaignData={campaignData} onInsertIntoNarration={onInsertIntoNarration} />
        </div>
      </div>
    </section>
  );
}
