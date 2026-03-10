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
  /** Optional pre-written text; when omitted, backend is expected to generate narration from context. */
  textOverride?: string;
  /** Voice to use for narration; when omitted, backend default will be used. */
  voice?: Pick<Voice, "id"> | null;
}

/**
 * Build a payload for narration from AI context.
 */
function buildNarrationPrompt(context: AiContext): string {
  const sceneTitle = context.scene?.title || "Untitled Scene";
  const locationName = context.location?.name;
  const npcNames = context.npcs.map((n) => n.name).join(", ");
  const recent = context.recentEvents.slice(-5).map((e) => `- [${e.type}] ${e.text}`).join("\n");

  return [
    `You are narrating a tabletop RPG scene: "${sceneTitle}".`,
    locationName ? `Location: ${locationName}.` : "",
    npcNames ? `NPCs present: ${npcNames}.` : "",
    recent ? `Recent events:\n${recent}` : "",
    "Write a short, vivid narration (2–5 sentences) that the GM can read aloud.",
  ]
    .filter(Boolean)
    .join("\n\n");
}

/**
 * Generate and speak narration for the current scene using AI context, returning text and an optional clip.
 * This is a thin service layer over /tts/narrate and the shared campaign context.
 */
export async function narrateCurrentScene(
  options: NarrateCurrentSceneOptions = {}
): Promise<NarrationResult> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const text = (options.textOverride || "").trim() || buildNarrationPrompt(context);
  const voiceId = options.voice?.id;

  const body: Record<string, unknown> = { text };
  if (voiceId) body.voice_id = voiceId;

  const res = await client.postNarrate(body);
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || "Failed to narrate current scene.");
  }

  const blob = await res.blob();
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

