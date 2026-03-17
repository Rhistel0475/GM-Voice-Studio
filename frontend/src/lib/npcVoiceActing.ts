/**
 * NPC Voice Acting Engine — generate dialogue with voice tone hints and optional audio.
 * Integrates with Voice Studio voices and campaign context.
 */

import { createClient } from "../api";
import { getAiContext, type AiContext } from "./aiContext";
import type { Npc } from "../types";

export interface NpcDialogueContext {
  sceneSummary?: string;
  recentEvents?: string[];
  locationName?: string;
  otherNpcNames?: string[];
}

export interface NpcDialogueResult {
  dialogue: string;
  toneHint?: string;
  audioUrl?: string;
}

export interface GenerateNpcDialogueOptions {
  apiKey?: string;
  /** Include TTS audio using NPC's assigned voice or fallback voiceId */
  generateAudio?: boolean;
  voiceId?: string;
  /** Override context; defaults to getAiContext() */
  context?: AiContext | null;
}

/**
 * Generate NPC dialogue with voice tone hints. Optionally produce TTS audio.
 */
export async function generateNpcDialogueWithVoice(
  npc: Npc,
  situationOrPrompt: string,
  options: GenerateNpcDialogueOptions = {}
): Promise<NpcDialogueResult> {
  const client = createClient(options.apiKey ?? "");
  const context = options.context ?? getAiContext();

  const payload = {
    npc_name: npc.name,
    personality: npc.summary ?? npc.role ?? "",
    faction: npc.factionId ?? "",
    situation: situationOrPrompt,
    conversation_history: [] as string[],
    tone_hint: npc.tags?.join(", ") ?? npc.role ?? undefined,
    scene_id: context.scene?.id,
    scene_summary: context.scene?.summary,
    location_name: context.location?.name,
    recent_events: context.recentEvents?.slice(-5).map((e) => e.text),
    scene_npcs: context.npcs?.map((entry) => entry.name),
    related_quests: context.relatedQuests?.map((quest) => String(quest.name || quest.title || "").trim()).filter(Boolean),
    codex_titles: context.codexReferences?.map((entry) => entry.title).filter(Boolean),
  };

  const res = await (client as { postAiDialogue: (body: unknown) => Promise<Response> }).postAiDialogue(payload);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "NPC dialogue generation failed.");
  }

  const data = (await res.json()) as { dialogue?: string; tone_hint?: string };
  const dialogue = data.dialogue ?? "";

  let audioUrl: string | undefined;
  if (options.generateAudio && dialogue) {
    const voiceId = npc.voiceId ?? options.voiceId;
    if (voiceId) {
      const ttsRes = await (client as { postNarrate: (body: { text: string; voice_id: string }) => Promise<Response> })
        .postNarrate({ text: dialogue, voice_id: voiceId });
      if (ttsRes.ok) {
        const blob = await ttsRes.blob();
        audioUrl = URL.createObjectURL(blob);
      }
    }
  }

  return {
    dialogue,
    toneHint: data.tone_hint,
    audioUrl,
  };
}
