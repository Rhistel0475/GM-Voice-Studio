import { createClient } from "../api";
import { getAiContext, type AiContext } from "./aiContext";
import type { ActionLogEvent } from "../types";

export interface GmAssistResponse {
  answer: string;
  reasoning?: string;
}

export interface GmAssistOptions {
  apiKey?: string;
}

function buildAssistPayload(query: string, context: AiContext) {
  return {
    mode: "gm_assist",
    query,
    campaign: context.campaign,
    session: context.session,
    scene: context.scene,
    npcs: context.npcs,
    location: context.location,
    relatedQuests: context.relatedQuests,
    recentEvents: context.recentEvents,
    codexReferences: context.codexReferences,
  };
}

/**
 * Real-time GM assist: answer lore/rules questions, suggest dialogue/reactions, summarize situation.
 */
export async function getGmAssistResponse(
  query: string,
  options: GmAssistOptions = {}
): Promise<GmAssistResponse> {
  const client = createClient(options.apiKey);
  const context = getAiContext();
  const body = buildAssistPayload(query, context);

  const res = await client.postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "GM Assist request failed.");
  }

  const data = await res.json();
  return {
    answer: data.answer ?? data.text ?? "",
    reasoning: data.reasoning ?? data.explanation,
  };
}

export interface SessionChronicle {
  sessionId: string;
  title: string;
  summary: string;
  majorEvents: string[];
  unresolvedHooks: string[];
  playerSafeSummary?: string;
}

/**
 * Generate a narrative chronicle for a session from logs, scenes, and memories.
 */
export async function generateSessionChronicle(
  sessionId: string,
  logs: ActionLogEvent[],
  options: GmAssistOptions = {}
): Promise<SessionChronicle> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const logLines = logs.map((e) => `[${e.type}] ${e.text}`).join("\n");

  const body = {
    mode: "session_chronicle",
    sessionId,
    campaign: context.campaign,
    session: context.session,
    scene: context.scene,
    npcs: context.npcs,
    relatedQuests: context.relatedQuests,
    codexReferences: context.codexReferences,
    recentEvents: context.recentEvents,
    rawLog: logLines,
  };

  const res = await client.postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to generate session chronicle.");
  }

  const data = await res.json();
  return {
    sessionId,
    title: data.title ?? context.session?.title ?? "Session Chronicle",
    summary: data.summary ?? "",
    majorEvents: data.majorEvents ?? [],
    unresolvedHooks: data.unresolvedHooks ?? [],
    playerSafeSummary: data.playerSafeSummary,
  };
}
