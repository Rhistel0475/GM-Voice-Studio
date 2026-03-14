import { createClient } from "../api";
import { getAiContext } from "./aiContext";

export interface SceneDirectorSuggestion {
  category: "tension" | "environment" | "npc_reaction" | "twist" | "pacing";
  text: string;
}

export interface SceneDirectorOptions {
  apiKey?: string;
}

const CATEGORY_KEYWORDS: Array<[SceneDirectorSuggestion["category"], RegExp]> = [
  ["tension",     /\b(tension|pressure|stakes|conflict|danger|dread|threat|escalat)\b/i],
  ["npc_reaction",/\b(npc|character|reaction|respond|ally|enemy|villain|guard|merchant)\b/i],
  ["twist",       /\b(twist|reveal|unexpected|surprise|secret|hidden|suddenly|discover)\b/i],
  ["pacing",      /\b(pacing|momentum|slow|fast|pause|hurry|rush|breathe|quiet)\b/i],
];

function inferCategory(text: string): SceneDirectorSuggestion["category"] {
  for (const [cat, re] of CATEGORY_KEYWORDS) {
    if (re.test(text)) return cat;
  }
  return "environment";
}

function buildQuery(context: ReturnType<typeof getAiContext>): string {
  const title = context.scene?.title || "Current Scene";
  const location = context.location?.name;
  const npcNames = context.npcs
    .map((n) => `${n.name}${n.role ? ` (${n.role})` : ""}`)
    .join(", ");
  const summary = context.scene?.summary?.slice(0, 250);
  const recent = context.recentEvents.slice(-3).map((e) => e.text).join(" · ");

  return [
    `As a Scene Director for a tabletop RPG, give 4–5 specific, actionable dramatic suggestions for the GM running this scene.`,
    `Scene: "${title}".`,
    location ? `Location: ${location}.` : null,
    npcNames ? `NPCs present: ${npcNames}.` : null,
    summary ? `Setup: ${summary}.` : null,
    recent ? `Recent events: ${recent}.` : null,
    `Include ideas for: tension escalation, environment details, NPC reactions, and one optional twist. Be brief and specific.`,
  ].filter(Boolean).join(" ");
}

export async function getSceneDirectorSuggestions(
  options: SceneDirectorOptions = {}
): Promise<SceneDirectorSuggestion[]> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const res = await client.postBrainQuery({ query: buildQuery(context) });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to get scene director suggestions.");
  }

  const data = await res.json();
  const text: string = (data.content || data.text || "").trim();
  if (!text) return [];

  // Parse each non-empty line into a structured suggestion
  return text
    .split("\n")
    .map((line) => line.replace(/^[\d]+[.)]\s*|^[-•*]\s*/, "").trim())
    .filter(Boolean)
    .slice(0, 5)
    .map((line): SceneDirectorSuggestion => ({
      category: inferCategory(line),
      text: line,
    }));
}
