import type { Npc, Voice } from "../types";
import { inferPresetForNpc, VOICE_PRESETS, type VoicePresetId } from "./voicePresets";

export interface VoiceSuggestion {
  presetId: VoicePresetId | null;
  presetName: string | null;
  reason: string;
  candidateVoices: Voice[];
}

/**
 * Suggest a voice (or set of voices) for an NPC based on role and traits.
 * This does not mutate state; UI can show suggestions and let the GM override.
 */
export function suggestVoiceForNpc(npc: Npc, voices: Voice[]): VoiceSuggestion {
  const presetId = inferPresetForNpc(npc);
  const preset = presetId ? VOICE_PRESETS[presetId] : null;

  const roleText = (npc.role || "").toLowerCase();
  const tags = (npc.tags || []).map((t) => t.toLowerCase());

  let reason = "Generic suggestion based on available voices.";
  if (preset) {
    reason = `Based on role/tags, "${preset.name}" fits ${npc.role || "this NPC"}.`;
  } else if (roleText) {
    reason = `Matching voices for role "${npc.role}".`;
  }

  const candidateVoices = voices.filter((v) => {
    const tone = (v.tone || "").toLowerCase();
    const name = (v.name || "").toLowerCase();
    const vTags = (v.tags || []).map((t) => t.toLowerCase());

    if (preset && (tone.includes(preset.name.toLowerCase()) || vTags.includes(preset.name.toLowerCase()))) {
      return true;
    }

    if (roleText && (name.includes(roleText) || tone.includes(roleText))) {
      return true;
    }

    if (tags.length && vTags.some((t) => tags.includes(t))) {
      return true;
    }

    return false;
  });

  return {
    presetId,
    presetName: preset?.name ?? null,
    reason,
    candidateVoices,
  };
}

