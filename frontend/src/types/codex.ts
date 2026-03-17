/**
 * Codex entry type discriminator.
 */
export type CodexEntryType = "document" | "location" | "rule" | "lore" | "faction";

/**
 * Codex entry model — campaign reference (location, rule, lore, etc.).
 */
export interface CodexEntry {
  id: string;
  campaignId: string;
  type: CodexEntryType;
  title: string;
  summary: string;
  content?: string;
  relatedNpcIds?: string[];
  relatedSceneIds?: string[];
  relatedNpcNames?: string[];
  relatedSceneNames?: string[];
  relatedLocationNames?: string[];
  relatedFactionNames?: string[];
  relatedQuestNames?: string[];
  tags?: string[];
  confidenceScore?: number;
  confidenceLabel?: "high" | "medium" | "low";
  needsReview?: boolean;
}
