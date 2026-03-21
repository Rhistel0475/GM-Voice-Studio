import { Outlet, useLocation } from "react-router-dom";
import { useCampaignContextStore } from "../store/campaignContext";
import { useActiveCampaign } from "../store/selectors";
import TopBanner from "../components/layout/TopBanner";
import SidebarNav from "../components/layout/SidebarNav";

const DEFAULT_BANNER_STATE = {
  sessionTime: "0:00",
  activeScene: "—",
  audioStatus: "idle",
};

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
  if (p === "/campaign") return "campaign";
  return "live";
}

export default function AppShell() {
  const requireApiKey = useCampaignContextStore((s) => s.requireApiKey);
  const apiKey = useCampaignContextStore((s) => s.apiKey);
  const setApiKey = useCampaignContextStore((s) => s.setApiKey);
  const activeCampaign = useActiveCampaign();
  const campaignDisplayName = activeCampaign?.title ?? activeCampaign?.name ?? "";

  const location = useLocation();
  const path = (location.pathname || "").replace(/^\/preview\/?/, "") || "/";
  const view = pathToView(path);
  const isPrepMode = view !== "live";
  const isLiveView = view === "live";

  const shellClassNames = [
    "dm-shell dm-fit mx-auto flex flex-col",
    isLiveView ? "min-h-screen overflow-hidden" : "min-h-0",
    isPrepMode ? "prep-mode" : "",
  ]
    .join(" ")
    .trim();

  const mainClassNames = [
    "app-stage flex-1 min-w-0 min-h-0",
    "p-3 md:p-4",
    isLiveView ? "overflow-hidden" : "overflow-x-hidden overflow-y-auto",
  ]
    .join(" ")
    .trim();

  return (
    <div className={shellClassNames}>
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
        campaignName={campaignDisplayName}
        activeScene={DEFAULT_BANNER_STATE.activeScene}
        sessionTime={DEFAULT_BANNER_STATE.sessionTime}
        audioStatus={DEFAULT_BANNER_STATE.audioStatus}
        nav={<SidebarNav inBanner />}
      />
      <div className="flex flex-1 min-h-0 gap-4 mt-1">
        <main className={mainClassNames}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
