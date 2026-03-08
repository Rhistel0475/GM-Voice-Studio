import React from "react";
import { EmptyState } from "../shared";

export default function NPCListSidebar({ npcs = [], selectedNpc, onSelectNpc }) {
  return (
    <div className="flex flex-col min-h-0">
      <p className="text-xs text-[var(--text-2)] mb-2">
        From campaign (Library parse). Click to load and assign voice.
      </p>
      <div className="overflow-auto flex-1 space-y-1">
        {npcs.length ? (
          npcs.map((n) => (
            <button
              key={n.name}
              type="button"
              className={`w-full text-left border p-2 rounded ${selectedNpc?.name === n.name ? "border-[var(--gold)] bg-[#1a1008]" : "border-[#5c3e23] bg-[#1a1008] hover:border-[var(--gold)]"}`}
              onClick={() => onSelectNpc(n)}
            >
              <span className="font-heading text-[var(--text-1)]">{n.name}</span>
              {n.role && <span className="text-xs text-[var(--text-2)] ml-2">— {n.role}</span>}
            </button>
          ))
        ) : (
          <EmptyState message="No NPCs in campaign. Use Library to parse adventure docs." />
        )}
      </div>
    </div>
  );
}
