import { BrowserRouter, Link, Navigate, Route, Routes } from "react-router-dom";
import { AppStateProvider } from "./context/AppStateContext";
import { CampaignProvider } from "./store/campaignContext";
import ImportPage from "./pages/ImportPage";
import PrepPage from "./pages/PrepPage";
import LiveBoardPage from "./pages/LiveBoardPage";

export default function App() {
  return (
    <CampaignProvider>
      {/* AppStateProvider: shared API key / legacy hooks for routes that still need it */}
      <AppStateProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-neutral-200 bg-white px-4 py-2">
            <nav className="flex gap-4 text-sm font-medium">
              <Link className="text-neutral-700 hover:text-neutral-900" to="/import">
                Import
              </Link>
              <Link className="text-neutral-700 hover:text-neutral-900" to="/prep">
                Prep
              </Link>
              <Link className="text-neutral-700 hover:text-neutral-900" to="/live">
                Live
              </Link>
            </nav>
          </header>
          <Routes>
            <Route path="/import" element={<ImportPage />} />
            <Route path="/prep" element={<PrepPage />} />
            <Route path="/live" element={<LiveBoardPage />} />
            <Route path="/" element={<Navigate to="/import" replace />} />
            <Route path="*" element={<Navigate to="/import" replace />} />
          </Routes>
        </div>
      </BrowserRouter>
      </AppStateProvider>
    </CampaignProvider>
  );
}
