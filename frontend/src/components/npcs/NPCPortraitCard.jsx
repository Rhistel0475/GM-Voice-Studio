import React from "react";

/**
 * Portrait area; shows image when portraitUrl is set, else placeholder.
 */
export default function NPCPortraitCard({ npc }) {
  if (!npc) return null;
  const name = npc.name || "Unknown";
  const portraitUrl = npc.portraitUrl || npc.portrait_url;
  return (
    <div className="portrait-card rounded overflow-hidden border-2 border-[#734f2c] bg-[#1a1008]">
      {portraitUrl ? (
        <img
          src={portraitUrl}
          alt={name}
          className="w-full aspect-square object-cover"
        />
      ) : (
        <div className="portrait-slot w-full aspect-square bg-[#0e0906] border-b border-[#5c3e23]" />
      )}
      <div className="px-2 py-1.5 border-t border-[#5c3e23]">
        <p className="font-heading text-[var(--gold)] text-sm text-center">{name}</p>
      </div>
    </div>
  );
}
