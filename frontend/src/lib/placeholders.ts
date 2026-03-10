import npcDefault from "../assets/placeholders/npcs/npc-default.svg";
import locationDefault from "../assets/placeholders/locations/location-default.svg";
import documentDefault from "../assets/placeholders/documents/document-default.svg";
import voiceDefault from "../assets/placeholders/voices/voice-default.svg";
import sceneDefault from "../assets/placeholders/scenes/scene-default.svg";
import partyDefault from "../assets/placeholders/party/party-default.svg";

type MaybeString = string | null | undefined;

function normalize(value: MaybeString): string {
  return (value || "").toLowerCase().trim();
}

export function getNpcPlaceholder(role?: MaybeString): string {
  const r = normalize(role);
  if (!r) return npcDefault;
  if (r.includes("bandit") || r.includes("thug") || r.includes("raider")) return npcDefault;
  if (r.includes("scout") || r.includes("ranger") || r.includes("hunter")) return npcDefault;
  if (r.includes("merchant") || r.includes("trader") || r.includes("innkeep")) return npcDefault;
  return npcDefault;
}

export function getLocationPlaceholder(kind?: MaybeString): string {
  const k = normalize(kind);
  if (!k) return locationDefault;
  if (k.includes("river") || k.includes("lake") || k.includes("harbor")) return locationDefault;
  if (k.includes("forest") || k.includes("wood") || k.includes("grove")) return locationDefault;
  if (k.includes("city") || k.includes("town") || k.includes("village")) return locationDefault;
  return locationDefault;
}

export function getVoicePlaceholder(tone?: MaybeString): string {
  const t = normalize(tone);
  if (!t) return voiceDefault;
  return voiceDefault;
}

export function getDocumentPlaceholder(type?: MaybeString): string {
  const t = normalize(type);
  if (!t) return documentDefault;
  return documentDefault;
}

export function getScenePlaceholder(kind?: MaybeString): string {
  const k = normalize(kind);
  if (!k) return sceneDefault;
  return sceneDefault;
}

export function getPartyPlaceholder(_: MaybeString = ""): string {
  return partyDefault;
}

