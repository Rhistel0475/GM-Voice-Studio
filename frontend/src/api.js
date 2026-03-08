/**
 * API client for GM Voice Studio backend.
 * Use relative URLs so they work with Vite proxy (dev) and same-origin (production).
 * Optional API key via createClient(apiKey) or per-request headers.
 */

const getBaseUrl = () => {
  if (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) {
    return import.meta.env.VITE_API_BASE.replace(/\/$/, "");
  }
  return "";
};

/**
 * Fetch /config (require_api_key, auto_query_on_voice, etc.).
 * @returns {Promise<{ require_api_key?: boolean, auto_query_on_voice?: boolean }>}
 */
export async function getConfig() {
  const base = getBaseUrl();
  const res = await fetch(`${base}/config`);
  return res.ok ? res.json() : { require_api_key: false, auto_query_on_voice: true };
}

/**
 * Create a client that sends X-API-Key on every request when apiKey is provided.
 * @param {string} [apiKey]
 * @returns API client with methods for voices, tts, brain, campaigns
 */
export function createClient(apiKey = "") {
  const base = getBaseUrl();
  const key = (apiKey || "").trim();

  const request = async (path, options = {}) => {
    const url = path.startsWith("http") ? path : `${base}${path}`;
    const headers = new Headers(options.headers || {});
    if (key) headers.set("X-API-Key", key);
    return fetch(url, { ...options, headers });
  };

  return {
    getBaseUrl: () => base,
    getConfig: () => getConfig(),

    getVoices: () => request("/voices").then((r) => r.json()),
    getVoiceList: () => request("/voices/list").then((r) => r.json()),
    getVoice: (id) => request(`/voices/${id}`).then((r) => r.json()),
    patchVoice: (id, body) =>
      request(`/voices/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    deleteVoice: (id) => request(`/voices/${id}`, { method: "DELETE" }),

    postTts: (formData) => request("/tts", { method: "POST", body: formData }),
    postNarrate: (body) =>
      request("/tts/narrate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),

    getJob: (jobId) => request(`/jobs/${jobId}`).then((r) => r.json()),
    getJobResult: (jobId) => request(`/jobs/${jobId}/result`),

    postBrainQuery: (body) =>
      request("/brain/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    postRagQuery: (body) =>
      request("/rag/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    postNpcGenerate: (body) =>
      request("/npc/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    postAiDialogue: (body) =>
      request("/ai/dialogue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),

    getCampaigns: () => request("/api/campaigns").then((r) => r.json()),
    getCampaign: (id) => request(`/api/campaigns/${id}`).then((r) => r.json()),
    deleteCampaign: (id) => request(`/api/campaigns/${id}`, { method: "DELETE" }),
  };
}

export { getBaseUrl };
