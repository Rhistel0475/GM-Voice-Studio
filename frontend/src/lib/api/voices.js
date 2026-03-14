/**
 * Voice Studio API layer. Fetches voices from /voices/list when available.
 * Returns empty arrays on failure — no mock data fallbacks.
 */

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
    provider: raw.provider,
    providerKind: raw.provider_kind || raw.providerKind,
    source: raw.source || "system",
    status: raw.status || "ready",
    featured: Boolean(raw.featured),
    icon: raw.icon,
    capabilities: raw.capabilities || {},
    deletable: typeof raw.deletable === "boolean" ? raw.deletable : undefined,
    editable: typeof raw.editable === "boolean" ? raw.editable : undefined,
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
  if (!authFetch) return [];
  try {
    const res = await authFetch("/voices/list");
    if (!res.ok) return [];
    const data = await res.json();
    const list = Array.isArray(data) ? data : data?.voices || data?.items || [];
    return list.map(mapVoiceToProfile);
  } catch {
    return [];
  }
}

/**
 * Delete a stored voice via DELETE /voices/:id.
 * @param {string} voiceId
 * @param {Function} authFetch
 * @returns {Promise<{ok: boolean, deleted?: string, error?: string}>}
 */
export async function deleteVoice(voiceId, authFetch) {
  if (!authFetch || !voiceId) return { ok: false, error: "Voice not found" };
  try {
    const res = await authFetch(`/voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
    const payload = await res.json().catch(() => ({}));
    if (res.ok) {
      return { ok: true, deleted: payload.deleted || voiceId };
    }
    return { ok: false, error: payload.detail || payload.error || res.statusText || "Delete failed" };
  } catch (e) {
    return { ok: false, error: e?.message || "Delete failed" };
  }
}

/**
 * Get generated audio clips from GET /voices/generated.
 * @param {Function} [authFetch]
 * @returns {Promise<import("../../types/voice").GeneratedAudio[]>}
 */
export async function getGeneratedAudio(authFetch) {
  if (!authFetch) return [];
  try {
    const res = await authFetch("/voices/generated");
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data : data?.clips || data?.items || [];
  } catch {
    return [];
  }
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
    return { ok: false, error: payload.detail || payload.error || `Server error ${res.status}` };
  } catch (e) {
    return { ok: false, error: e?.message || "Network error" };
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
