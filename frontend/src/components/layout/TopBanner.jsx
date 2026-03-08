import React from "react";

export default function TopBanner({ campaignName, activeScene, sessionTime, audioStatus }) {
  const audioLabel =
    audioStatus === "loading" ? "Loading…" : audioStatus === "playing" ? "Playing" : "Idle";
  return (
    <header className="dm-header">
      <div className="header-glow" />
      <div className="relative z-10 flex flex-col items-center gap-1">
        <h1 className="font-heading text-[clamp(1.6rem,2.25vw,2.85rem)] leading-[1.05] text-[#e7c27a] drop-shadow-[0_2px_1px_#1a0f08]">
          GM Voice Studio
        </h1>
        <p className="font-heading text-[clamp(1.1rem,1.7vw,2.1rem)] leading-[1.05] text-[#d9b878]">
          {campaignName ? `Active Campaign: ${campaignName}` : "No Campaign Loaded"}
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4 mt-2 text-sm font-heading">
          <span className="text-[var(--text-2)]">
            <span className="text-[var(--gold)] uppercase tracking-wider">Session</span> {sessionTime ?? "0:00"}
          </span>
          <span className="text-[var(--text-2)]">
            <span className="text-[var(--gold)] uppercase tracking-wider">Scene</span> {activeScene ?? "—"}
          </span>
          <span className="text-[var(--text-2)]">
            <span className="text-[var(--gold)] uppercase tracking-wider">Audio</span> {audioLabel}
          </span>
        </div>
      </div>
    </header>
  );
}
