import type { Npc, Voice } from "../types";

const SUPPORTED_TAGS = [
  "male",
  "female",
  "old",
  "young",
  "rough",
  "noble",
  "merchant",
  "villain",
  "guard",
  "scholar",
] as const;

type SupportedTag = (typeof SUPPORTED_TAGS)[number];

const TAG_KEYWORDS: Record<SupportedTag, string[]> = {
  male: ["male", "man", "boy", "gentleman", "sir"],
  female: ["female", "woman", "girl", "lady", "madam"],
  old: ["old", "elder", "elderly", "aged", "ancient", "veteran"],
  young: ["young", "youth", "teen", "child", "apprentice"],
  rough: ["rough", "gruff", "raspy", "harsh", "gravelly", "scarred"],
  noble: ["noble", "lord", "lady", "duke", "duchess", "regal", "refined", "courtier", "aristocrat"],
  merchant: ["merchant", "trader", "vendor", "shopkeeper", "innkeeper", "barkeep", "apothecary"],
  villain: ["villain", "tyrant", "cruel", "evil", "menacing", "bandit", "assassin", "cultist", "warlord"],
  guard: ["guard", "captain", "soldier", "watch", "warden", "marshal", "knight", "mercenary"],
  scholar: ["scholar", "sage", "professor", "academic", "librarian", "scribe", "researcher", "historian"],
};

const NPC_FIELD_WEIGHTS: Array<{ key: string; weight: number }> = [
  { key: "role", weight: 5 },
  { key: "summary", weight: 3 },
  { key: "description", weight: 3 },
  { key: "personality", weight: 3 },
  { key: "name", weight: 1 },
];

export interface VoiceSuggestionCandidate extends Voice {
  confidence?: number;
  matchedTags?: string[];
}

export interface VoiceSuggestion {
  confidence: number;
  matchedTags: string[];
  candidateVoices: VoiceSuggestionCandidate[];
}

function normalizeText(value: unknown): string {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function containsKeyword(text: string, keyword: string): boolean {
  const pattern = new RegExp(`(?:^|\\b)${keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:\\b|$)`);
  return pattern.test(text);
}

function inferNpcTagWeights(npc: Npc): Map<SupportedTag, number> {
  const weights = new Map<SupportedTag, number>();
  const source = npc as Npc & Record<string, unknown>;

  for (const tag of SUPPORTED_TAGS) {
    let score = 0;
    for (const field of NPC_FIELD_WEIGHTS) {
      const text = normalizeText(source[field.key]);
      if (text && TAG_KEYWORDS[tag].some((keyword) => containsKeyword(text, keyword))) {
        score += field.weight;
      }
    }
    if (score > 0) weights.set(tag, score);
  }

  return weights;
}

function deriveVoiceTags(voice: Voice): Set<SupportedTag> {
  const explicit = (voice.tags || [])
    .map((tag) => normalizeText(tag))
    .filter(Boolean);
  const search = normalizeText([voice.name, voice.tone, voice.accent, ...voice.tags].join(" "));
  const tags = new Set<SupportedTag>();

  for (const tag of SUPPORTED_TAGS) {
    if (explicit.includes(tag) || TAG_KEYWORDS[tag].some((keyword) => containsKeyword(search, keyword))) {
      tags.add(tag);
    }
  }

  return tags;
}

function confidenceForMatch(matchedTags: SupportedTag[], npcWeights: Map<SupportedTag, number>): number {
  const total = Array.from(npcWeights.values()).reduce((sum, value) => sum + value, 0);
  if (total <= 0) return 0.1;
  const matched = matchedTags.reduce((sum, tag) => sum + (npcWeights.get(tag) || 0), 0);
  if (matched <= 0) return 0.1;
  return Math.min(0.99, Math.max(0.1, Number((matched / total).toFixed(3))));
}

/**
 * Suggest one or more voice-library candidates for an NPC using voice tags.
 */
export function suggestVoiceForNpc(npc: Npc, voices: Voice[]): VoiceSuggestion {
  const npcWeights = inferNpcTagWeights(npc);

  const candidateVoices = [...voices]
    .map((voice) => {
      const voiceTags = deriveVoiceTags(voice);
      const matchedTags = SUPPORTED_TAGS.filter((tag) => npcWeights.has(tag) && voiceTags.has(tag));
      const weightedScore = matchedTags.reduce((sum, tag) => sum + (npcWeights.get(tag) || 0), 0);
      return {
        ...voice,
        confidence: confidenceForMatch(matchedTags, npcWeights),
        matchedTags,
        weightedScore,
      };
    })
    .sort((a, b) => {
      if (b.weightedScore !== a.weightedScore) return b.weightedScore - a.weightedScore;
      if ((b.tags?.length || 0) !== (a.tags?.length || 0)) return (b.tags?.length || 0) - (a.tags?.length || 0);
      return (a.name || "").localeCompare(b.name || "");
    })
    .map(({ weightedScore: _weightedScore, ...voice }) => voice);

  return {
    confidence: candidateVoices[0]?.confidence || 0.1,
    matchedTags: candidateVoices[0]?.matchedTags || [],
    candidateVoices,
  };
}
