import React from "react";

/**
 * Placeholder for NPC portrait image; can be extended with image URL later.
 */
export default function NPCPortraitCard({ npc }) {
  if (!npc) return null;
  return (
    <div className="portrait-card rounded overflow-hidden">
      <div className="portrait-slot w-full" />
      <p className="font-heading text-[var(--text-1)] text-center text-sm mt-1">{npc.name}</p>
    </div>
  );
}
