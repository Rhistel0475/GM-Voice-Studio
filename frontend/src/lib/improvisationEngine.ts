/**
 * AI Improvisation Engine — generate unexpected story developments (new NPC, complication, conflict, clue).
 */

import { createClient } from "../api";
import { getAiContext, type AiContext } from "./aiContext";

export type ImprovisationCategory =
  | "new_npc_arrival"
  | "environmental_complication"
  | "sudden_conflict"
  | "new_clue_discovery";

export interface ImprovisationResult {
  category: ImprovisationCategory;
  title: string;
  description: string;
  suggestedNpc?: { name: string; role?: string };
  suggestedClue?: string;
  suggestedConflict?: string;
}

export interface GenerateImprovisationOptions {
  apiKey?: string;
  context?: AiContext | null;
}

/**
 * Generate an unexpected story development for the current scene.
 */
export async function generateImprovisation(
  options: GenerateImprovisationOptions = {}
): Promise<ImprovisationResult> {
  const client = createClient(options.apiKey ?? "");
  const context = options.context ?? getAiContext();

  const body = {
    mode: "improvisation",
    campaign: context.campaign,
    session: context.session,
    scene: context.scene,
    npcs: context.npcs,
    location: context.location,
    recentEvents: context.recentEvents,
    codexReferences: context.codexReferences,
  };

  const res = await (client as { postBrainQuery: (body: unknown) => Promise<Response> }).postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Improvisation generation failed.");
  }

  const data = (await res.json()) as {
    category?: string;
    title?: string;
    description?: string;
    suggestedNpc?: { name: string; role?: string };
    suggestedClue?: string;
    suggestedConflict?: string;
  };

  return {
    category: (data.category as ImprovisationCategory) ?? "environmental_complication",
    title: data.title ?? "Unexpected development",
    description: data.description ?? "",
    suggestedNpc: data.suggestedNpc,
    suggestedClue: data.suggestedClue,
    suggestedConflict: data.suggestedConflict,
  };
}
