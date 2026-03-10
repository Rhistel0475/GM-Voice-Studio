# React frontend (primary preview UI)

The React app in `frontend/` is the **primary preview UI** for GM Voice Studio. The shared campaign data layer (types, Zustand store, selectors) lives in TypeScript (`src/types/*.ts`, `src/store/*.ts`); UI components can remain JS/JSX and consume the store via hooks. It is built into `static/frontend/` and served by FastAPI at `/preview` (and `/preview/` for SPA subpaths).

## UI framework and design system

The app uses a shared layout and fantasy-themed design system so all product views (Live Board, Codex, NPC Workshop, Voice Studio) feel consistent.

### Layout

- **AppShell** (`src/layout/AppShell.jsx`) — Wraps the app: TopBanner, SidebarNav, and main content (Outlet).
- **TopBanner** — Campaign name, active scene, session timer, audio status.
- **SidebarNav** — Nav links for Live Board, Codex, NPC Workshop, Voice Studio, Prep, Library, Settings. On tablet (md) the sidebar collapses to icons only; on desktop (xl) it shows labels.
- **WorkspaceContainer** — Optional wrapper for page content.
- **WorkspaceHeader** — Page title with optional icon (emoji or Lucide) and subtitle; use on each workspace view.

### Shared components (`src/components/shared/`)

- **ParchmentCard** — Card with optional title, headerAction, children, footer. Uses parchment styling.
- **FantasyButton** — Variants: primary, secondary, ghost, danger.
- **SectionHeader** — Subhead bar with optional icon, title, subtitle, action.
- **StatusPill** — Status labels: ready, recording, generating, playing, offline, saved, training, failed.
- **SearchInput** — Shared search field (chat-input style) for Codex, NPC list, Voice library.
- **EmptyState** — Message and optional icon/action when a list is empty.
- **LoadingState** — Spinner and message.
- **AudioPlayer** — Stub play/pause control and status; wire to audio source when used.
- **ModalShell** — Overlay modal with title and close.

### Theme

- **Palette** — `src/styles/theme.css` defines: `--background-dark`, `--parchment`, `--gold`, `--wood`, `--accent`. Main `styles.css` imports it and adds wood gradients, panel styles, and typography (Cinzel for headings, Crimson Text / Inter for body).

### Routes

Routes are defined in `App.jsx` (React Router). Paths: `/` (Live Board), `/codex`, `/npcs`, `/voices`, `/settings`, `/prep`, `/intake`. The shell renders an `<Outlet />`; `CurrentView` resolves the path and renders the corresponding page.

**App feature entries** (`src/app/`) — Each product view has an entry under `app/`:

- `app/codex/` — Codex (campaign research)
- `app/npcs/` — NPC Workshop
- `app/voices/` — Voice Studio
- `app/live-board/` — Live Board

Each exports the page component used by the router; feature-specific components remain under `components/` (e.g. `components/codex/`, `components/npcs/`, `components/voices/`, `components/live-board/`).

---

## Quick start

```bash
# From project root
cd frontend
npm install
npm run build
```

Then start the backend and open **http://localhost:7862/preview** (or your `PORT`).

## Build output

- **Output directory:** `static/frontend/` (relative to repo root). FastAPI serves this via `static/frontend/index.html` when you request `/preview`.
- **Base path:** The app is built with `base: "/preview/"` so assets load at `/preview/assets/...`.
- **One-command build from root:** Run `./scripts/build-frontend.sh` (or `make frontend` if you add a Makefile). This runs `npm run build` inside `frontend/` and ensures the result is in `static/frontend`.

## Development server

For frontend-only iteration with hot reload and API proxy:

```bash
cd frontend
npm run dev
```

- Dev server runs at **http://localhost:5173** (or next available port).
- Vite proxies `/api`, `/adventure`, `/ai`, `/rag`, `/brain`, `/tts`, `/voices`, `/npc`, `/campaign-assets`, `/static`, and `/ws` to `http://localhost:7862`. Start the backend on 7862 so API and WebSocket work.
- After changing the frontend, run `npm run build` again to update `static/frontend` for production or for testing with the real server.

## API client

The frontend can use the shared API client in `frontend/src/api.js`:

- **`getBaseUrl()`** — Returns the API base URL (empty string when using relative URLs behind the same host or proxy).
- **`getConfig()`** — Fetches `GET /config` (e.g. `require_api_key`, `auto_query_on_voice`).
- **`createClient(apiKey?)`** — Returns an object with methods that send requests with optional `X-API-Key` header:
  - `getConfig()`, `getVoices()`, `getVoiceList()`, `postTts(formData)`, `postNarrate(body)`, `postBrainQuery(body)`, `getCampaigns()`, `getCampaign(id)`, `deleteCampaign(id)`, etc.

Use relative paths (e.g. `/voices/list`) so they work with the dev proxy and with production when the app is served from the same origin as the API.

## Environment

- **Dev proxy:** No env needed; Vite proxies to `localhost:7862`.
- **Production:** The app is served from the same host as the API (e.g. `http://localhost:7862/preview` and `http://localhost:7862/tts`). For a different API origin, set `VITE_API_BASE` at build time and use it in the API client (see `api.js`).

## Smoke test

After building, start the backend and open http://localhost:7862/preview. Confirm: config loads, voice list loads, narrate and brain query work when the backend is configured.

See [Smoke Test Checklist](smoke-test-checklist.md) for full steps.
