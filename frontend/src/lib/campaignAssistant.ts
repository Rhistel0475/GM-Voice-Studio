import { createClient } from "../api";
import { getAiContext } from "./aiContext";

export interface CampaignAssistantAnswer {
  answer: string;
  sources?: string[];
}

export interface CampaignAssistantOptions {
  apiKey?: string;
}

export async function askCampaignAssistant(
  question: string,
  options: CampaignAssistantOptions = {}
): Promise<CampaignAssistantAnswer> {
  const client = createClient(options.apiKey);
  const context = getAiContext();

  const body = {
    mode: "campaign_assistant",
    question,
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
    throw new Error(err || "Campaign assistant request failed.");
  }

  const data = await res.json();
  return {
    answer: data.answer ?? data.text ?? "",
    sources: data.sources ?? [],
  };
}

