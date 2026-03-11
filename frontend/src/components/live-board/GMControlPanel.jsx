import { useState } from "react";
import QuickToolGrid from "./QuickToolGrid";
import { QUICK_TOOLS } from "./constants";
import { ModalShell } from "../shared";
import { getPartyPlaceholder } from "../../lib/placeholders";

const DEFAULT_PARTY = [
  { name: "ATHELRED", hp: "—", ac: "—" },
  { name: "SERAPINA", hp: "—", ac: "—" },
  { name: "BORWIN", hp: "—", ac: "—" },
  { name: "DUNCAN", hp: "—", ac: "—" },
];

export default function GMControlPanel({ campaignData, scene, selectedNpcName, onSelectNpc }) {
  const [activeTool, setActiveTool] = useState(null);
  const party = campaignData?.party?.length ? campaignData.party : DEFAULT_PARTY;
  const allNpcs = campaignData?.npcs || [];
  const sceneNpcs = (scene?.npcs || []).map((npcName) => allNpcs.find((n) => n.name === npcName)).filter(Boolean);
  const allItems = campaignData?.items || [];
  const sceneItems = (scene?.items || []).map((itemName) => allItems.find((i) => i.name === itemName)).filter(Boolean);
  const allReveals = campaignData?.reveals || [];
  const sceneReveals = (scene?.reveals || []).map((revealName) => allReveals.find((r) => r.name === revealName)).filter(Boolean);

  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
        <div className="panel-head">
          <div className="plaque">Quick Tools</div>
        </div>
        <div className="panel-body min-h-0">
          <QuickToolGrid tools={QUICK_TOOLS} onToolClick={setActiveTool} />
        </div>
      </section>

      <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
        <div className="panel-head">
          <div className="plaque">Active Scene</div>
        </div>
        <div className="panel-body space-y-2 overflow-y-auto min-h-[120px] max-h-[220px]">
          {sceneNpcs.map((npc) => (
            <button
              key={npc.name}
              type="button"
              className={`tracker-row w-full text-left cursor-pointer ${selectedNpcName === npc.name ? "is-active border-[#d4af37]" : ""}`}
              onClick={() => onSelectNpc(npc.name)}
            >
              <span className="encounter-name">{npc.name}</span>
              <span className="text-[#9b7440] text-xs">{npc.role}</span>
            </button>
          ))}
          {sceneItems.map((item) => (
            <div key={item.name} className="tracker-row w-full text-left">
              <span className="encounter-name">{item.name}</span>
              <span className="text-[#9b7440] text-xs">Item</span>
            </div>
          ))}
          {sceneReveals.map((reveal) => (
            <div key={reveal.name} className="tracker-row w-full text-left">
              <span className="encounter-name">{reveal.name}</span>
              <span className="text-[#9b7440] text-xs capitalize">{reveal.type}</span>
            </div>
          ))}
          {sceneNpcs.length === 0 && sceneItems.length === 0 && sceneReveals.length === 0 && (
            <div className="intake-empty text-xs">No NPCs, items, or reveals in this scene.</div>
          )}
        </div>
      </section>

      <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden flex-shrink-0">
        <div className="panel-head">
          <div className="plaque">Party Roster</div>
        </div>
        <div className="panel-body">
          <div className="grid grid-cols-2 gap-1">
            {party.slice(0, 4).map((char) => {
              const portraitSrc = getPartyPlaceholder();
              return (
                <article key={char.name} className="party-card">
                  <div
                    className="party-face"
                    style={{ backgroundImage: `url(${portraitSrc})`, backgroundSize: "cover", backgroundPosition: "center" }}
                  />
                <div className="party-meta">
                  <div className="name text-sm">{char.name}</div>
                  <div className="stats text-xs">
                    {char.hp !== "—" ? (
                      <>HP {char.hp} <span>/ AC {char.ac}</span></>
                    ) : (
                      <span className="text-[#6b5030]">—</span>
                    )}
                  </div>
                </div>
              </article>
              );
            })}
          </div>
        </div>
      </section>

      {activeTool && (
        <ModalShell title={activeTool.name} onClose={() => setActiveTool(null)}>
          <p className="text-[#e8c787] font-heading text-lg mb-4">
            {activeTool.name} Module
          </p>
          <p className="text-[#b88b46] text-sm italic">
            Interface coming soon. This will hook into our backend API for live generation.
          </p>
        </ModalShell>
      )}
    </div>
  );
}
