import { createClient } from "../api";
import { getAiContext } from "./aiContext";
import type { Session } from "../types";

export interface PlayerRecap {
  sessionId: string;
  shortRecap: string;
  fullRecap: string;
}

export interface NextSessionTeaser {
  sessionId: string;
  teaser: string;
}

export interface RecapOptions {
  apiKey?: string;
  /** When true, allow GM-only info to appear in the recap. Default false. */
  includeGmSecrets?: boolean;
}

export async function generatePlayerRecap(
  session: Session,
  options: RecapOptions = {}
): Promise<PlayerRecap> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const body = {
    mode: "player_recap",
    session,
    campaign: context.campaign,
    scene: context.scene,
    recentEvents: context.recentEvents,
    codexReferences: context.codexReferences,
    includeGmSecrets: !!options.includeGmSecrets,
  };

  const res = await client.postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to generate player recap.");
  }

  const data = await res.json();
  return {
    sessionId: session.id,
    shortRecap: data.shortRecap ?? "",
    fullRecap: data.fullRecap ?? "",
  };
}

export async function generateNextSessionTeaser(
  session: Session,
  options: RecapOptions = {}
): Promise<NextSessionTeaser> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const body = {
    mode: "next_session_teaser",
    session,
    campaign: context.campaign,
    scene: context.scene,
    recentEvents: context.recentEvents,
    codexReferences: context.codexReferences,
  };

  const res = await client.postBrainQuery(body);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to generate next session teaser.");
  }

  const data = await res.json();
  return {
    sessionId: session.id,
    teaser: data.teaser ?? "",
  };
}

