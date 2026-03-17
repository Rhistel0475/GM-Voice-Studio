import { createClient } from "../api";
import { getAiContext, type AiContext } from "./aiContext";
import type { NarrationClip, Voice } from "../types";
import { createId, nowIso } from "./utils/ids";

export interface NarrationResult {
  text: string;
  clip?: NarrationClip;
}

export interface NarrateCurrentSceneOptions {
  apiKey?: string;
  /**
   * Pre-written text to speak directly (no Claude generation step).
   * When omitted, Claude is queried first to generate narration from AI context.
   */
  textOverride?: string;
  /** Voice to use for narration; when omitted, backend default will be used. */
  voice?: Pick<Voice, "id"> | null;
}

/**
 * Build a brain-query prompt for narration from AI context.
 * Returns a query string that will be sent to /brain/query; the response content
 * becomes the narration text passed to /tts/narrate.
 */
function buildNarrationQuery(context: AiContext): string {
  const sceneTitle = context.scene?.title || "Untitled Scene";
  const locationName = context.location?.name;
  const npcNames = context.npcs.map((n) => n.name).join(", ");
  const recent = context.recentEvents.slice(-3).map((e) => e.text).join(" · ");
  const sceneSummary = context.scene?.summary;
  const questSummary = context.relatedQuests
    .map((quest) => String(quest.name || quest.title || "").trim())
    .filter(Boolean)
    .slice(0, 3)
    .join(", ");
  const codexHints = context.codexReferences
    .map((entry) => entry.title)
    .filter(Boolean)
    .slice(0, 3)
    .join(", ");

  const parts = [
    `Write a short, vivid 2–4 sentence scene narration for the GM to read aloud.`,
    `Scene: "${sceneTitle}".`,
    locationName ? `Location: ${locationName}.` : null,
    sceneSummary ? `Context: ${sceneSummary.slice(0, 150)}.` : null,
    npcNames ? `NPCs present: ${npcNames}.` : null,
    questSummary ? `Related quests: ${questSummary}.` : null,
    codexHints ? `Relevant campaign knowledge: ${codexHints}.` : null,
    recent ? `Recent events: ${recent}.` : null,
    `Return only the narration text with no preamble or commentary.`,
  ];

  return parts.filter(Boolean).join(" ");
}

/**
 * Generate and speak narration for the current scene.
 *
 * Two-step flow (when no textOverride is given):
 *   1. Call /brain/query with a narration prompt → get generated narration text
 *   2. Call /tts/narrate with that text → get audio blob
 *
 * When textOverride is given, step 1 is skipped and the text is spoken directly.
 *
 * Returns { text, clip } where clip contains an object-URL for the audio.
 */
export async function narrateCurrentScene(
  options: NarrateCurrentSceneOptions = {}
): Promise<NarrationResult> {
  const client = createClient(options.apiKey);
  const context = getAiContext();
  const voiceId = options.voice?.id;

  // ── Step 1: Resolve narration text ────────────────────────────────────────
  let text: string;
  if (options.textOverride?.trim()) {
    text = options.textOverride.trim();
  } else {
    const query = buildNarrationQuery(context);
    const genRes = await client.postBrainQuery({ query });
    if (!genRes.ok) {
      const errText = await genRes.text();
      throw new Error(errText || "Failed to generate narration text.");
    }
    const genData = await genRes.json();
    text = (genData.content || genData.text || "").trim();
    if (!text) throw new Error("AI returned empty narration.");
  }

  // ── Step 2: Speak via TTS ─────────────────────────────────────────────────
  const ttsBody: Record<string, unknown> = { text };
  if (voiceId) ttsBody.voice_id = voiceId;

  const ttsRes = await client.postNarrate(ttsBody);
  if (!ttsRes.ok) {
    const errText = await ttsRes.text();
    throw new Error(errText || "Failed to narrate scene.");
  }

  const blob = await ttsRes.blob();
  const audioUrl = URL.createObjectURL(blob);

  const clip: NarrationClip = {
    id: createId("clip"),
    campaignId: context.campaign?.id ?? "",
    sessionId: context.session?.id,
    sceneId: context.scene?.id,
    title: context.scene?.title || "Scene Narration",
    voiceId,
    audioUrl,
    createdAt: nowIso(),
  };

  return { text, clip };
}
