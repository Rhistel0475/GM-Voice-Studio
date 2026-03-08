/**
 * Voice Studio API layer. Fetches voices from /voices/list when available;
 * falls back to mock data. Generated audio and jobs use mock until backend is ready.
 */

import {
  MOCK_VOICE_PROFILES,
  MOCK_GENERATED_AUDIO,
  MOCK_VOICE_CLONE_JOB,
} from "../utils/mockData";

/**
 * Map backend voice item to VoiceProfile shape.
 * @param {Object} raw - Item from /voices/list
 * @returns {import("../../types/voice").VoiceProfile}
 */
function mapVoiceToProfile(raw) {
  const id = raw.id || raw.voice_id || raw.name || "unknown";
  return {
    id: String(id),
    voice_id: String(id),
    name: raw.name || raw.display_name || "Unnamed Voice",
    source: raw.source || "system",
    status: raw.status || "ready",
    accent: raw.accent,
    tone: raw.tone,
    tags: Array.isArray(raw.tags) ? raw.tags : [],
    sampleUrl: raw.sample_url || raw.sampleUrl,
    assignedNPCs: raw.assigned_npcs || raw.assignedNPCs || [],
    description: raw.description,
    updatedAt: raw.updated_at || raw.updatedAt,
  };
}

/**
 * Get voices for Voice Studio. Tries /voices/list; falls back to mock data.
 * @param {Function} authFetch - From AppStateContext
 * @returns {Promise<import("../../types/voice").VoiceProfile[]>}
 */
export async function getVoices(authFetch) {
  if (!authFetch) return MOCK_VOICE_PROFILES;
  try {
    const res = await authFetch("/voices/list");
    if (!res.ok) return MOCK_VOICE_PROFILES;
    const data = await res.json();
    const list = Array.isArray(data) ? data : data?.voices || data?.items || [];
    return list.length ? list.map(mapVoiceToProfile) : MOCK_VOICE_PROFILES;
  } catch {
    return MOCK_VOICE_PROFILES;
  }
}

/**
 * Get generated audio clips. Uses mock data until backend endpoint exists.
 * @param {Function} [authFetch]
 * @returns {Promise<import("../../types/voice").GeneratedAudio[]>}
 */
export async function getGeneratedAudio(authFetch) {
  // Future: GET /voices/generated or /api/generated-audio
  return Promise.resolve(MOCK_GENERATED_AUDIO);
}

/**
 * Submit voice clone job. Uses existing POST /voices/clone when authFetch provided.
 * @param {FormData} formData - name, sample file(s), etc.
 * @param {Function} authFetch
 * @returns {Promise<{ job_id?: string, voice_id?: string, ok: boolean }>}
 */
export async function submitClone(formData, authFetch) {
  if (!authFetch) {
    return Promise.resolve({ ok: false });
  }
  try {
    const res = await authFetch("/voices/clone", { method: "POST", body: formData });
    const payload = await res.json().catch(() => ({}));
    if (res.ok && (payload.job_id || payload.voice_id)) {
      return { ok: true, job_id: payload.job_id, voice_id: payload.voice_id };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

/**
 * Poll clone job status. Uses GET /jobs/:id when authFetch provided.
 * @param {string} jobId
 * @param {Function} authFetch
 * @returns {Promise<import("../../types/voice").VoiceCloneJob | null>}
 */
export async function getCloneJobStatus(jobId, authFetch) {
  if (!authFetch || !jobId) return null;
  try {
    const res = await authFetch(`/jobs/${jobId}`);
    if (!res.ok) return null;
    const data = await res.json();
    return {
      id: data.id || jobId,
      status: data.status || "processing",
      progress: typeof data.progress === "number" ? data.progress : 0,
      stepLabel: data.step_label || data.stepLabel,
      voice_id: data.voice_id,
    };
  } catch {
    return null;
  }
}

/**
 * Stub: return a mock job for demo when no real job id.
 * @returns {import("../../types/voice").VoiceCloneJob}
 */
export function getMockCloneJob() {
  return { ...MOCK_VOICE_CLONE_JOB };
}
