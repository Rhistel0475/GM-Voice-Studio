# GM Voice Studio — Implementation Gap Report

**Generated:** 2025-03-07  
**Scope:** Frontend codebase vs. stated goals (LiveBoard command center, Codex research view, NPC Workshop, Voice Studio, ingestion pipeline, campaign context architecture).

---

## 1. Complete

- **Campaign context store (`store/campaignContext.ts`)**  
  Zustand store with campaigns, sessions, scenes, NPCs, voices, codex entries, action log, narration clips. Actions: setActiveCampaign/Session/Scene, assignVoiceToNpc, assignNpcToScene, addCodexEntryToScene, addActionLogEvent, addNarrationClip, upserts. Implemented and used.

- **Selectors (`store/selectors.ts`)**  
  Pure selectors and hook wrappers for active campaign/session/scene, scene NPCs, scene codex entries, action log for active scene, narration clips, codex/NPCs per campaign, voices per NPC. Complete.

- **AI context (`lib/aiContext.ts`)**  
  `buildAiContext(state, options)` and `getAiContext()` / `useAiContext()` build structured context (campaign, session, scene, npcs, location, recentEvents, codexReferences) from store. Used by narration and assist layers.

- **LiveBoard layout and panels**  
  `LiveBoardPage` implements 12-column grid (2–5–2–3): GM Control (left), Live Session (center), NPC Presence (right-of-center), Codex (right). Viewport-locked layout in AppShell for live view. All four zones present and wired.

- **Codex research view structure**  
  Three-column layout: `CodexSidebar` (campaign selector, search, filters, tags), `CodexResultList`, `CodexDetailPanel`. Filter by query, category, tags; select item; detail with “Add to Live Board” when CampaignContext exists.

- **NPC Workshop creation view**  
  Three-column: roster sidebar (search, filters, list), Create/Edit (generator form with streaming, voice picker, play sample), Preview card (save, push to Live Board, voice assignment). Uses campaign store when available; save/push call API or fallback to local state.

- **Voice Studio media view**  
  Two-column: library (search, filters, grid/list, generated audio list), detail + clone wizard + narration textarea. Clone flow with file upload, job polling, preview, save. Assign voice to NPC via campaign context; narration uses selected voice and `/tts/narrate`.

- **Extraction review queue store (`store/extractionReview.ts`)**  
  Zustand store: enqueueBatch, updateItemStatus, editItemEntity, removeItem, clearQueue, setAutoApproveHighConfidence. Confidence → initial status (high + auto-approve → approved; medium/low → needs_review). Types in `types/extraction.ts` (ExtractionReviewItem, ExtractionEntity, etc.).

- **Document ingestion client API**  
  `lib/documentIngestion.ts`: `ingestAdventureDocument(file, apiKey)` calls `postAdventureParse(formData)`; `extractStructuredSections(documentText)` for client-side section parsing. API layer in `api.js` exposes `postAdventureParse`.

- **Supporting lib modules (present and typed)**  
  Relationship resolution, campaign world summary, encounter manager, campaign memory, timeline, recaps, GM assist, campaign assistant, scene director, AI narration, voice presets, voice suggestions, session logger, scene transition, one-click encounter, liveboard campaign context, etc. Implemented as specified in phase prompts.

---

## 2. Partially Complete

- **LiveBoard command center**
  - **Complete:** Layout, GM Control (Quick Tools, Active Scene, Party Roster), NPC Presence, Codex quick reference, Session Log area, Co-DM query input, mic/wake/auto-query toggles, scene selector, narrate button.
  - **Partial:** Quick Tools open a “coming soon” modal (no dice, grimoire, map, loot, encounter backend). Center column uses **legacy** narration: `handleNarrate` sends `scene.read_aloud` to `/tts/narrate` (no AI-generated narration from context). `liveboardCampaignContext.generateSceneNarration` and `aiNarration.narrateCurrentScene` exist but are **not** wired from the LiveBoard UI. GM Assist and Scene Director services exist but have **no** UI entry points (no “Ask GM Assist” or “Scene Director suggestions” in the center panel). Session log displays action log but Co-DM submit path and backend integration are legacy (no guaranteed use of `addSessionLogEntry` from liveboardCampaignContext for all flows).

- **Codex research view**
  - **Complete:** Sidebar, result list, detail panel, filters, campaign selector, getCodexItems from campaign + store merge.
  - **Partial:** Items are derived from **legacy** campaign shape (scenes, npcs, locations) or mock data; no dedicated backend codex API. “Add to Live Board” uses `addCodexEntryToScene(entry?.id ?? entry?.title, campaignCtx.activeSceneIndex)` — mixing id/title and scene index may be fragile. No RAG or rules lookup beyond placeholder.

- **NPC Workshop**
  - **Complete:** Roster, filters, generator (streaming), preview, save, push to Live Board, voice assignment (store + API when available).
  - **Partial:** Voice **suggestions** (`suggestVoiceForNpc`, `inferPresetForNpc`) are **not** used in the Workshop UI. Regenerate backstory is a stub (“coming soon”). Save/push fall back to local state when backend returns null; no extraction review integration (new NPCs from ingestion don’t flow through review queue UI).

- **Voice Studio**
  - **Complete:** Library, filters, grid/list, clone wizard, narration panel, assign to NPC, play sample/clip.
  - **Partial:** Voice **presets** and **suggestions** (voicePresets, voiceSuggestions) are **not** surfaced in the UI (no preset dropdown or “suggest for NPC”). Unassign NPC is “coming soon.” Generated audio list and play use `/tts/narrate` with clip title/voiceId (no persisted clip URLs). No “reuse for narration” wiring into LiveBoard.

- **Ingestion pipeline**
  - **Complete:** Backend parse endpoints used from Intake (`/adventure/parse`, `/adventure/ai-parse`), image extraction, save to campaign (setCampaignData). Extraction types and review **store** (extractionReview) with confidence/status.
  - **Partial:** Intake does **not** use `lib/documentIngestion.ingestAdventureDocument`; it uses inline `runParse(url)` and form handling in App.jsx. No staged pipeline UI (normalize → chunk → classify → extract → relate → review). **No review queue UI**: `useExtractionReviewQueueStore` is never used by any component; extracted entities are not enqueued or shown for approve/edit/reject. Relationship resolution and campaign world are lib-only (not triggered from ingestion flow).

- **Campaign context architecture**
  - **Complete:** Store + selectors, CampaignProvider, deriveLegacyCampaign/deriveLegacyActiveScene/deriveLegacyActionLog for backward compatibility, useCampaignOptional, LiveBoard and Codex/NPC/Voice pages can use store when provider is present.
  - **Partial:** **Dual data paths**: LiveBoard (and others) still receive `campaignData` from AppState and merge with `campaignCtx` (`campaign = campaignCtx?.campaign ?? campaignData`). Prep/Intake and parts of App.jsx operate on `campaignData`/setCampaignData; store is hydrated from seed and optionally from Intake “Save to Campaign” but there is no single “save to store” path from Intake. Backend persistence for store mutations is stubbed (TODOs: assignVoiceToNpc, addActionLogEvent, etc.). CampaignProvider’s `setCampaign` is a no-op.

---

## 3. Missing

- **Extraction review queue UI**  
  No component that renders `useExtractionReviewQueueStore().items`, shows entity type/confidence/source/status, or provides approve / edit / reject. Review queue is never populated from parse results.

- **Ingestion → store/review pipeline**  
  Parse results (npcs, scenes, locations, codex_entries, etc.) are not converted to `ExtractionBatchResult` and passed to `enqueueBatch`. No flow: upload → parse → extract → review queue → approve → write to campaign store (or codex/NPC APIs).

- **LiveBoard AI narration and GM assist UI**  
  No button or panel that calls `generateSceneNarration()` / `narrateCurrentScene()` for context-based narration. No UI that calls `getGmAssistResponse()` or `getSceneDirectorSuggestions()` or displays their results.

- **Voice presets and suggestions in UI**  
  No preset selector in Voice Studio or NPC Workshop; no “suggest voice” using `suggestVoiceForNpc` / `inferPresetForNpc`.

- **Unified campaign write path**  
  No single path that takes Intake parse result (or review-approved entities) and updates the campaign context store (and optionally backend). Intake “Save to Campaign” updates AppState only.

- **Backend persistence for campaign context**  
  Assignments and session log are in-memory only; no implemented PATCH/POST for voice assignment, action log, or codex-to-scene links.

- **Quick Tools implementations**  
  Dice, Grimoire, Map, Loot Table, Encounter open a placeholder modal only.

---

## 4. Fragile or Risky Areas

- **Dual campaign data (AppState vs. store)**  
  LiveBoard, Codex, NPC Workshop, and Voice Studio all accept `campaignData` from props/AppState and optionally merge with CampaignContext. Inconsistent use (e.g., Codex `onAddToLiveBoard` using `entry?.id ?? entry?.title` and scene index) can cause subtle bugs. Refactors that assume “store only” or “campaignData only” could break one path.

- **CampaignContext legacy mapping**  
  `deriveLegacyCampaign` maps store scenes/NPCs to legacy shape (e.g. `scene.npcs` as names, `scene.codexRefs`). If the store uses different semantics (e.g. codexEntryIds vs. codexRefs), components that expect legacy shape can misbehave. `assignNpcToScene` in context takes npcNameOrId; NPC Workshop sometimes passes name, which depends on store NPCs having matching names.

- **Codex “Add to Live Board”**  
  Passing `entry?.id ?? entry?.title` and `activeSceneIndex` mixes id and title; if CodexItem uses title as id in some paths, linking may be wrong. No validation that the entry exists in the store.

- **Intake parse vs. documentIngestion**  
  Two ways to parse: App.jsx `runParse("/adventure/parse")` (and ai-parse) vs. `documentIngestion.ingestAdventureDocument`. DocumentIngestion is unused; future changes to one may not align with the other (e.g. response shape, error handling).

- **API fallbacks and stubs**  
  saveNPC, pushToLiveBoard, getCodexItems, getNPCs return null or mock data when backend is missing. Components that assume “saved” or “pushed” may still show local-only state; toasts say “Saved locally (no backend)” but store may not be updated consistently for all code paths.

- **Large App.jsx**  
  LiveBoard, Prep, Intake, MiddleColumn, and routing live in one file (2400+ lines). Changes to one view or data flow risk regressions elsewhere; hard to test in isolation.

- **Scene/npc identity**  
  Some flows use scene/NPC by index, others by id; legacy shapes use `scene.npcs` as name strings. Store uses ids (sceneId, npcIds). Any mix of index vs. id vs. name in the same flow is a source of bugs.

---

## 5. Next 5 Implementation Priorities (in order)

1. **Wire extraction review queue into ingestion**  
   - After parse (Quick Parse or AI Parse) in Intake, map parse result to `ExtractionBatchResult` (entities with confidence/source) and call `enqueueBatch`.  
   - Add a Review Queue UI (e.g. in Intake or a dedicated tab) that lists `useExtractionReviewQueueStore().items`, shows entity type, confidence, source, status, and provides approve / edit / reject.  
   - On approve, persist entities into campaign store or existing APIs (npcs, scenes, codex) so Codex and NPC Workshop see them.  
   - This closes the pipeline gap and makes ingestion usable for feeding the rest of the app.

2. **Connect LiveBoard to AI narration and GM assist**  
   - In the Live Session center column, add a “Generate narration” (or “AI narrate”) action that calls `generateSceneNarration()` / `narrateCurrentScene()` and displays/plays the result, and optionally adds a clip to the store.  
   - Add an “Ask GM Assist” (or similar) input that calls `getGmAssistResponse(query)` and shows the answer in the session area.  
   - Optionally add “Scene Director” suggestions (e.g. a collapsible panel) that calls `getSceneDirectorSuggestions()` and displays tension/environment/NPC suggestions.  
   - Use `addSessionLogEntry` from liveboardCampaignContext for Co-DM and narration events so the session log stays in sync with the store.

3. **Single campaign write path from Intake**  
   - Define a single “Apply to campaign” path: either “Save to Campaign” writes to the campaign context store (and optionally syncs to backend), or parse → review → approve writes into the store.  
   - Ensure Intake “Save to Campaign” (and any review-approved batch) updates the store (campaigns, scenes, npcs, codex entries) so LiveBoard, Codex, and NPC Workshop read from one source of truth when CampaignProvider is used.  
   - Reduce reliance on setCampaignData for critical campaign structure; prefer store as source of truth and derive legacy campaign for components that still need it.

4. **Voice suggestions and presets in UI**  
   - In NPC Workshop (e.g. in the voice dropdown or next to it), call `suggestVoiceForNpc(selectedNpc, voices)` and show a “Suggested: …” or preset badge; allow one-click apply.  
   - In Voice Studio, expose voice presets (e.g. dropdown or tags) using `VOICE_PRESETS` / `applyVoicePreset` for playback or metadata so GMs can pick by tone/style.  
   - Ensures the existing voice suggestion and preset logic is discoverable and used.

5. **Backend persistence for campaign context**  
   - Implement or wire backend for: (a) voice-to-NPC assignment (PATCH npc or PATCH campaign), (b) action log / session events (POST session/events or WebSocket), (c) codex-to-scene links if stored server-side.  
   - Keep store as source of truth and sync on success; show clear errors when backend fails.  
   - Enables multi-tab or multi-device consistency and avoids loss of assignments and log on refresh.

---

*Report produced from static analysis of the frontend codebase; no code was modified.*
