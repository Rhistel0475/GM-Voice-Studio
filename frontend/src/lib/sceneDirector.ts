import { createClient } from "../api";
import { getAiContext } from "./aiContext";

export interface SceneDirectorSuggestion {
  category: "tension" | "environment" | "npc_reaction" | "twist" | "pacing";
  text: string;
}

export interface SceneDirectorOptions {
  apiKey?: string;
}

type SceneDirectorSuggestionWire = {
  category?: SceneDirectorSuggestion["category"] | string;
  text?: string;
};

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
  const raw: unknown[] = Array.isArray(data?.suggestions) ? data.suggestions : [];

  return raw.map((s, index): SceneDirectorSuggestion => {
    const obj = (s && typeof s === "object" ? (s as SceneDirectorSuggestionWire) : null);
    const category =
      obj?.category === "tension" ||
      obj?.category === "environment" ||
      obj?.category === "npc_reaction" ||
      obj?.category === "twist" ||
      obj?.category === "pacing"
        ? obj.category
        : "environment";
    const text = obj?.text ? obj.text : typeof s === "string" ? s : `Suggestion ${index + 1}`;
    return { category, text };
  });
}

