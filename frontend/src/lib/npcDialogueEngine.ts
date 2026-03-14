/**
 * NPC Dialogue Engine for LiveBoard — personality, role, scene, environment, recent events.
 * UI: Talk, Rumor, Threat, Hint.
 */

import { useCampaignContextStore } from "../store/campaignContext";
import { getAiContext, type AiContext } from "./aiContext";
import { useSceneStateStore } from "./sceneStateEngine";
import { generateNpcDialogueWithVoice } from "./npcVoiceActing";
import { addSessionLogEntry } from "./liveboardCampaignContext";
import type { Npc } from "../types";

export type NpcDialogueType = "talk" | "rumor" | "threat" | "hint";

export interface DialogueContext {
  npc: Npc;
  sceneSummary?: string;
  locationName?: string;
  environmentState: string;
  recentEvents: string[];
  otherNpcNames: string[];
}

/**
 * Build dialogue context for an NPC (for use in generateNpcDialogue or UI).
 */
export function buildDialogueContext(npcId: string, options?: { context?: AiContext | null }): DialogueContext | null {
  const state = useCampaignContextStore.getState();
  const npc = state.npcs.find((n) => n.id === npcId) ?? null;
  if (!npc) return null;

  const context = options?.context ?? getAiContext();
  const sceneState = useSceneStateStore.getState();

  return {
    npc,
    sceneSummary: context.scene?.summary,
    locationName: context.location?.name,
    environmentState: sceneState.environmentState,
    recentEvents: context.recentEvents.slice(-10).map((e) => e.text),
    otherNpcNames: context.npcs.filter((n) => n.id !== npcId).map((n) => n.name),
  };
}

const TYPE_PROMPTS: Record<NpcDialogueType, string> = {
  talk: "The NPC is speaking directly to the party. Generate natural in-character dialogue.",
  rumor: "The NPC shares a rumor or hearsay. Generate something they might have heard.",
  threat: "The NPC makes a threat or hostile remark. Keep it game-appropriate.",
  hint: "The NPC gives a subtle hint or clue about the situation or world.",
};

/**
 * Generate NPC dialogue of the given type (talk, rumor, threat, hint).
 */
export async function generateNpcDialogue(
  npcId: string,
  type: NpcDialogueType,
  options?: { apiKey?: string }
): Promise<{ dialogue: string; toneHint?: string }> {
  const ctx = buildDialogueContext(npcId);
  if (!ctx) throw new Error(`NPC not found: ${npcId}`);

  const situation = `${TYPE_PROMPTS[type]}\nScene: ${ctx.sceneSummary ?? "—"}. Location: ${ctx.locationName ?? "—"}. Environment: ${ctx.environmentState}.`;
  const result = await generateNpcDialogueWithVoice(ctx.npc, situation, {
    ...options,
    generateAudio: false,
  });
  return { dialogue: result.dialogue, toneHint: result.toneHint };
}

/**
 * Generate TTS audio for NPC dialogue text using the NPC's assigned voice.
 */
export async function generateNpcDialogueAudio(
  npcId: string,
  text: string,
  options?: { apiKey?: string }
): Promise<string | null> {
  const state = useCampaignContextStore.getState();
  const npc = state.npcs.find((n) => n.id === npcId);
  if (!npc) return null;

  const result = await generateNpcDialogueWithVoice(npc, text, {
    ...options,
    textOverride: text,
    generateAudio: true,
  });
  return result.audioUrl ?? null;
}

/**
 * Record NPC dialogue in the session log.
 */
export function recordNpcDialogue(npcId: string, text: string): void {
  const state = useCampaignContextStore.getState();
  const npc = state.npcs.find((n) => n.id === npcId);
  const label = npc ? `${npc.name}: ` : "";
  addSessionLogEntry({ type: "npc", text: `${label}${text}` });
}
