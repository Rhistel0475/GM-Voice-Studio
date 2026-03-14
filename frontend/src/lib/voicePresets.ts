import type { Voice, Npc } from "../types";

export type VoicePresetId =
  | "warm_narrator"
  | "grim_veteran"
  | "noble_scholar"
  | "whispering_oracle"
  | "rough_bandit";

export interface VoicePreset {
  id: VoicePresetId;
  name: string;
  tone: string;
  pacing: "slow" | "medium" | "fast";
  intensity: "soft" | "medium" | "strong";
  style: string;
}

export const VOICE_PRESETS: Record<VoicePresetId, VoicePreset> = {
  warm_narrator: {
    id: "warm_narrator",
    name: "Warm Narrator",
    tone: "warm, inviting, storyteller",
    pacing: "medium",
    intensity: "soft",
    style: "cozy fantasy narrator with clear diction",
  },
  grim_veteran: {
    id: "grim_veteran",
    name: "Grim Veteran",
    tone: "gravelly, world-weary, low register",
    pacing: "medium",
    intensity: "strong",
    style: "battle-scarred soldier with dry humor",
  },
  noble_scholar: {
    id: "noble_scholar",
    name: "Noble Scholar",
    tone: "refined, educated, measured",
    pacing: "slow",
    intensity: "soft",
    style: "aristocratic sage or court chronicler",
  },
  whispering_oracle: {
    id: "whispering_oracle",
    name: "Whispering Oracle",
    tone: "soft, echoing, otherworldly",
    pacing: "slow",
    intensity: "soft",
    style: "mystical seer speaking in half-whispers",
  },
  rough_bandit: {
    id: "rough_bandit",
    name: "Rough Bandit",
    tone: "raspy, aggressive, mocking",
    pacing: "fast",
    intensity: "strong",
    style: "roadside brigand with coarse language",
  },
};

/**
 * Apply a voice preset to a Voice model, returning a new object.
 * This does not persist; callers should PATCH via the API or store.
 */
export function applyVoicePreset(voice: Voice, preset: VoicePreset): Voice {
  const tags = new Set(voice.tags || []);
  tags.add(preset.name);
  return {
    ...voice,
    tone: preset.tone,
    tags: Array.from(tags),
  };
}

/**
 * Heuristic: pick a reasonable preset id for an NPC based on role/tags.
 */
export function inferPresetForNpc(npc: Pick<Npc, "role" | "tags">): VoicePresetId | null {
  const role = (npc.role || "").toLowerCase();
  const tagText = (npc.tags || []).join(" ").toLowerCase();
  const text = `${role} ${tagText}`;

  if (text.includes("merchant") || text.includes("innkeep") || text.includes("barkeep")) {
    return "warm_narrator";
  }
  if (text.includes("bandit") || text.includes("thug") || text.includes("raider")) {
    return "rough_bandit";
  }
  if (text.includes("noble") || text.includes("lord") || text.includes("lady") || text.includes("scholar")) {
    return "noble_scholar";
  }
  if (text.includes("oracle") || text.includes("seer") || text.includes("prophet")) {
    return "whispering_oracle";
  }
  if (text.includes("veteran") || text.includes("captain") || text.includes("commander")) {
    return "grim_veteran";
  }

  return null;
}

