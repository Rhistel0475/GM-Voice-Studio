import type { CampaignSystemPreset } from "./system";

export interface CampaignKnowledgeRecord {
  id?: string;
  name?: string;
  title?: string;
  description?: string;
  summary?: string;
  content?: string;
  objective?: string;
  stakes?: string;
  type?: string;
  tags?: string[];
  related_npcs?: string[];
  related_locations?: string[];
  related_scenes?: string[];
  confidence?: number;
  confidence_score?: number;
  confidence_label?: string;
  needs_review?: boolean;
  review_priority?: string;
  [key: string]: unknown;
}

/**
 * Campaign model for the shared data layer.
 * Backend: GET/PATCH /api/campaigns/:id
 */
export interface Campaign {
  id: string;
  name: string;
  setting?: string;
  activeSessionId?: string;
  systemId?: string;
  system?: CampaignSystemPreset;
  quests?: CampaignKnowledgeRecord[];
  factions?: CampaignKnowledgeRecord[];
  lore?: CampaignKnowledgeRecord[];
  relationships?: Array<Record<string, unknown>>;
  reviewSummary?: Record<string, number>;
}
