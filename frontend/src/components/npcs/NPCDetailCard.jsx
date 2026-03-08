import React from "react";
import { ParchmentCard } from "../shared";
import { EmptyState } from "../shared";

export default function NPCDetailCard({ npc }) {
  if (!npc) {
    return <EmptyState message="Select an NPC to view details." />;
  }
  return (
    <ParchmentCard title={npc.name}>
      {npc.role && <p className="text-sm text-[var(--text-2)] mb-1">{npc.role}</p>}
      {npc.personality && (
        <div className="text-sm text-[var(--ink-1)] whitespace-pre-wrap">{npc.personality}</div>
      )}
      {npc.faction && <p className="text-xs text-[var(--text-2)] mt-2">Faction: {npc.faction}</p>}
    </ParchmentCard>
  );
}
