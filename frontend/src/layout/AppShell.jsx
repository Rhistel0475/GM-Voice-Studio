import React from "react";
import { Outlet, useLocation } from "react-router-dom";
import { useAppState } from "../context/AppStateContext";
import TopBanner from "../components/layout/TopBanner";
import SidebarNav from "../components/layout/SidebarNav";

function pathToView(path) {
  let p = (path || "/").replace(/\/+$/, "").replace(/^\/preview\/?/, "").trim() || "/";
  if (!p.startsWith("/")) p = `/${p}`;
  if (p === "/" || p === "") return "live";
  if (p === "/codex") return "codex";
  if (p === "/npcs") return "npc-workshop";
  if (p === "/voices") return "voice-studio";
  if (p === "/settings") return "settings";
  if (p === "/prep") return "prep";
  if (p === "/intake") return "intake";
  return "live";
}

export default function AppShell() {
  const { campaignData, bannerState, requireApiKey, apiKey, setApiKey } = useAppState();
  const location = useLocation();
  const path = (location.pathname || "").replace(/^\/preview\/?/, "") || "/";
  const view = pathToView(path);
  const isPrepMode = view !== "live";

  return (
    <div className={`dm-shell dm-fit mx-auto flex flex-col min-h-0 ${isPrepMode ? "prep-mode" : ""}`}>
      {requireApiKey && (
        <div className="mx-auto mb-2 w-full max-w-sm rounded border border-[#6f4a27] bg-[#1a0f06] p-2 text-xs text-[#d2ab68]">
          <label className="block mb-1 font-heading text-[#e7c27a]">API Key Required</label>
          <input
            type="password"
            className="w-full bg-[#120a04] border border-[#4f341f] text-[#e8c887] rounded px-2 py-1"
            placeholder="Enter API key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </div>
      )}
      <TopBanner
        campaignName={campaignData?.title}
        activeScene={bannerState.activeScene}
        sessionTime={bannerState.sessionTime}
        audioStatus={bannerState.audioStatus}
      />
      <div className="flex flex-1 min-h-0 gap-4 mt-1">
        <aside className="flex-shrink-0 w-40 md:w-14 xl:w-48 panel-ornate rounded overflow-auto">
          <SidebarNav />
        </aside>
        <main className="app-stage flex-1 min-w-0 min-h-0 overflow-x-hidden overflow-y-auto p-3 md:p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
