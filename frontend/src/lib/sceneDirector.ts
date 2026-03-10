import { createClient } from "../api";
import { getAiContext } from "./aiContext";

export interface SceneDirectorSuggestion {
  category: "tension" | "environment" | "npc_reaction" | "twist" | "pacing";
  text: string;
}

export interface SceneDirectorOptions {
  apiKey?: string;
}

export async function getSceneDirectorSuggestions(
  options: SceneDirectorOptions = {}
): Promise<SceneDirectorSuggestion[]> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const body = {
    mode: "scene_director",
    campaign: context.campaign,
    session: context.session,
    scene: context.scene,
    npcs: context.npcs,
    location: context.location,
    recentEvents: context.recentEvents,
    codexReferences: context.codexReferences,
  };

  const res = await client.postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to get scene director suggestions.");
  }

  const data = await res.json();
  const raw = Array.isArray(data.suggestions) ? data.suggestions : [];

  return raw.map((s: any, index: number): SceneDirectorSuggestion => ({
    category: s.category || "environment",
    text: s.text || String(s) || `Suggestion ${index + 1}`,
  }));
}

