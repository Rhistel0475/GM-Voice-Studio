import React from "react";
import SectionHeader from "../components/layout/SectionHeader";
import GMControlPanel from "../components/live-board/GMControlPanel";
import SessionStream from "../components/live-board/SessionStream";
import CodexSidePanel from "../components/live-board/CodexSidePanel";

/**
 * Live Board: central command center for the GM during live gameplay.
 * Three-zone layout: Left = GM Control (quick tools, scene, party) | Center = Session stream | Right = Codex.
 * Banner (campaign, scene, timer, audio) is provided by AppShell TopBanner.
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
    <section className="live-board min-h-0 flex flex-col gap-4">
      <div className="min-h-0 flex-1 grid grid-cols-1 xl:grid-cols-12 gap-4 xl:gap-5">
        {/* LEFT: GM Control Panel — quick tools, active scene, party roster */}
        <aside className="xl:col-span-3 min-h-0 flex flex-col gap-2">
          <SectionHeader title="GM Control" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden">
            <GMControlPanel
              campaignData={campaignData}
              scene={scene}
              selectedNpcName={selectedNpcName}
              onSelectNpc={onSelectNpc}
            />
          </div>
        </aside>

        {/* CENTER: Session stream — SessionLog, NarrationComposer, AudioPlaybackCard (via middleColumn) */}
        <main className="xl:col-span-5 min-h-0 flex flex-col gap-2">
          <SectionHeader title="Live Session" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden">
            <SessionStream>{middleColumn}</SessionStream>
          </div>
        </main>

        {/* RIGHT: Codex — Documents, NPCs, Locations, Rules tabs + quick preview cards */}
        <aside className="xl:col-span-4 min-h-0 flex flex-col gap-2">
          <SectionHeader title="Codex" className="rounded-t-lg" />
          <div className="min-h-0 flex-1 min-w-0 overflow-hidden">
            <CodexSidePanel
              campaignData={campaignData}
              onInsertIntoNarration={onInsertIntoNarration}
            />
          </div>
        </aside>
      </div>
    </section>
  );
}
