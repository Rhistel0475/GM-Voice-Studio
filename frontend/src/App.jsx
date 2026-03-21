import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { CampaignProvider } from "./store/CampaignProvider";
import LibraryPage from "./pages/LibraryPage";
import PrepPage from "./pages/PrepPage";
import LiveBoardPage from "./pages/LiveBoardPage";

export default function App() {
  return (
    <CampaignProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col app-root-nav">
          <header className="app-top-nav border-b px-4 py-2">
            <nav className="flex gap-4 text-sm font-medium font-heading">
              <NavLink
                className={({ isActive }) => `app-top-nav-link${isActive ? " is-active" : ""}`}
                to="/live"
              >
                Live
              </NavLink>
              <NavLink
                className={({ isActive }) => `app-top-nav-link${isActive ? " is-active" : ""}`}
                to="/prep"
              >
                Prep
              </NavLink>
              <NavLink
                className={({ isActive }) => `app-top-nav-link${isActive ? " is-active" : ""}`}
                to="/library"
              >
                Library
              </NavLink>
            </nav>
          </header>
          <Routes>
            <Route path="/live" element={<LiveBoardPage />} />
            <Route path="/prep" element={<PrepPage />} />
            <Route path="/library" element={<LibraryPage />} />
            <Route path="/" element={<Navigate to="/live" replace />} />
            <Route path="*" element={<Navigate to="/live" replace />} />
          </Routes>
        </div>
      </BrowserRouter>
    </CampaignProvider>
  );
}
