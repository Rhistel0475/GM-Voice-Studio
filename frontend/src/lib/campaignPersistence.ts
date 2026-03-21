/**
 * Campaign persistence helpers — sync store mutations to the backend.
 *
 * Strategy: store is the immediate source of truth (optimistic).
 * Backend calls are fire-and-forget with silent failure logging.
 * Backend campaign ID is stored in localStorage so it survives page reloads.
 */

const BACKEND_ID_KEY = "gm_campaign_backend_id";

/** Return the backend integer campaign ID stored after parse/load, or null. */
export function getBackendCampaignId(): number | null {
  try {
    const raw = localStorage.getItem(BACKEND_ID_KEY);
    if (!raw) return null;
    const id = parseInt(raw, 10);
    return Number.isFinite(id) ? id : null;
  } catch {
    return null;
  }
}

/** Persist the backend integer campaign ID after parse or load. */
export function setBackendCampaignId(id: number | string | null | undefined): void {
  try {
    if (id == null || id === "") {
      localStorage.removeItem(BACKEND_ID_KEY);
    } else {
      localStorage.setItem(BACKEND_ID_KEY, String(id));
    }
  } catch {
    // localStorage unavailable
  }
}

type AuthFetch = (url: string, options?: RequestInit) => Promise<Response>;

/**
 * Persist a voice-to-NPC assignment to the backend.
 * Fire-and-forget: resolves immediately; backend failure is logged but not thrown.
 */
export async function persistNpcVoice(
  authFetch: AuthFetch,
  npcName: string,
  voiceId: string | null | undefined
): Promise<void> {
  const campaignId = getBackendCampaignId();
  if (!campaignId || !npcName) return;
  try {
    await authFetch(`/api/campaigns/${campaignId}/npcs/${encodeURIComponent(npcName)}/voice`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_id: voiceId ?? "" }),
    });
  } catch {
    // Backend unavailable — store is source of truth
  }
}

/**
 * Persist scene read-aloud / GM notes to the backend (PATCH).
 * Matches scene by title within the stored backend campaign id.
 */
export async function persistSceneContent(
  authFetch: AuthFetch,
  sceneTitle: string,
  patch: { readAloud?: string; notes?: string }
): Promise<void> {
  const campaignId = getBackendCampaignId();
  const title = (sceneTitle || "").trim();
  if (!campaignId || !title) return;
  const body: Record<string, string> = {};
  if (patch.readAloud !== undefined) body.read_aloud = patch.readAloud;
  if (patch.notes !== undefined) body.notes = patch.notes;
  if (Object.keys(body).length === 0) return;
  try {
    await authFetch(
      `/api/campaigns/${campaignId}/scenes/${encodeURIComponent(title)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
  } catch {
    // Backend unavailable — store is source of truth
  }
}

export interface SessionEventPayload {
  type: string;
  text: string;
  scene_id?: string | null;
  session_id?: string | null;
  id?: string;
  created_at?: string;
}

/**
 * Persist a session event (action log entry) to the backend.
 * Fire-and-forget.
 */
export async function persistSessionEvent(
  authFetch: AuthFetch,
  event: SessionEventPayload
): Promise<void> {
  const campaignId = getBackendCampaignId();
  if (!campaignId || !event.text) return;
  try {
    await authFetch(`/api/campaigns/${campaignId}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
    });
  } catch {
    // Backend unavailable — store is source of truth
  }
}

/**
 * Load persisted session events from the backend for the current campaign.
 * Returns an empty array on failure.
 */
export async function loadSessionEvents(
  authFetch: AuthFetch,
  sceneId?: string,
  limit = 100
): Promise<SessionEventPayload[]> {
  const campaignId = getBackendCampaignId();
  if (!campaignId) return [];
  try {
    const params = new URLSearchParams({ limit: String(limit) });
    if (sceneId) params.set("scene_id", sceneId);
    const res = await authFetch(`/api/campaigns/${campaignId}/events?${params}`);
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.events) ? data.events : [];
  } catch {
    return [];
  }
}
