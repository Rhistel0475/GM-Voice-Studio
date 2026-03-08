import React from "react";
import { FantasyButton } from "../shared";

const SECTIONS = ["Campaign", "Adventures", "Locations", "NPCs", "Lore", "Rules"];

export default function CodexSidebar({
  section,
  onSectionChange,
  scenes = [],
  locations = [],
  npcs = [],
  selectedDoc,
  onSelectDoc,
}) {
  return (
    <div className="flex flex-col gap-2 overflow-auto h-full">
      <div className="plaque mb-2">Sections</div>
      <div className="flex flex-col gap-1">
        {SECTIONS.map((sec) => (
          <FantasyButton
            key={sec}
            variant={section === sec ? "primary" : "secondary"}
            className="text-left w-full"
            onClick={() => {
              onSectionChange(sec);
              onSelectDoc(null);
            }}
          >
            {sec}
          </FantasyButton>
        ))}
      </div>
      {section === "Adventures" && (
        <div className="mt-2 space-y-1 overflow-auto">
          {scenes.map((s, i) => (
            <button
              key={i}
              type="button"
              className="w-full text-left border border-[#5c3e23] bg-[#1a1008] p-2 text-sm text-[var(--text-1)] hover:border-[var(--gold)]"
              onClick={() => onSelectDoc(s)}
            >
              {s.title || `Scene ${i + 1}`}
            </button>
          ))}
          {!scenes.length && <div className="intake-empty text-xs">No scenes. Use Library to parse.</div>}
        </div>
      )}
      {section === "Locations" && (
        <div className="mt-2 space-y-1 overflow-auto">
          {locations.map((loc, i) => (
            <button
              key={i}
              type="button"
              className="w-full text-left border border-[#5c3e23] bg-[#1a1008] p-2 text-sm text-[var(--text-1)] hover:border-[var(--gold)]"
              onClick={() => onSelectDoc(typeof loc === "string" ? { title: loc, body: "" } : loc)}
            >
              {typeof loc === "string" ? loc : loc.name || loc}
            </button>
          ))}
          {!locations.length && <div className="intake-empty text-xs">No locations.</div>}
        </div>
      )}
      {section === "NPCs" && (
        <div className="mt-2 space-y-1 overflow-auto">
          {npcs.map((n) => (
            <button
              key={n.name}
              type="button"
              className="w-full text-left border border-[#5c3e23] bg-[#1a1008] p-2 text-sm text-[var(--text-1)] hover:border-[var(--gold)]"
              onClick={() => onSelectDoc(n)}
            >
              {n.name}
            </button>
          ))}
          {!npcs.length && <div className="intake-empty text-xs">No NPCs.</div>}
        </div>
      )}
    </div>
  );
}
