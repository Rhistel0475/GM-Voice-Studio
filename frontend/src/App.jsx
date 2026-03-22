import { Component } from "react";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { CampaignProvider } from "./store/CampaignProvider";
import LibraryPage from "./pages/LibraryPage";
import PrepPage from "./pages/PrepPage";
import LiveBoardPage from "./pages/LiveBoardPage";

class AppErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 16, fontFamily: "sans-serif", maxWidth: 600 }}>
          <h2 style={{ color: "#c53030" }}>Something went wrong</h2>
          <pre style={{ overflow: "auto", background: "#1a1a1a", color: "#f0f0f0", padding: 12, borderRadius: 6 }}>
            {this.state.error?.message}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            style={{ marginTop: 12, padding: "8px 16px", cursor: "pointer" }}
          >
            Dismiss
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <CampaignProvider>
      <BrowserRouter basename="/preview">
        <AppErrorBoundary>
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
        </AppErrorBoundary>
      </BrowserRouter>
    </CampaignProvider>
  );
}
