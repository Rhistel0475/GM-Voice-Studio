/**
 * Ambient scene audio — play/stop background sounds per scene type.
 * Increases immersion during live sessions.
 */

import type { Scene } from "../types";

export type SceneAmbientType =
  | "forest"
  | "tavern"
  | "storm"
  | "dungeon"
  | "city_market"
  | "none";

/** Placeholder or CDN URLs per ambient type. Replace with your assets or backend URLs. */
const AMBIENT_URLS: Record<Exclude<SceneAmbientType, "none">, string> = {
  forest: "",   // e.g. /assets/ambient/forest.mp3
  tavern: "",
  storm: "",
  dungeon: "",
  city_market: "",
};

let activeAudio: HTMLAudioElement | null = null;

function inferAmbientType(scene: Scene): SceneAmbientType {
  const text = [
    scene.title,
    scene.summary,
    (scene.tags ?? []).join(" "),
  ]
    .join(" ")
    .toLowerCase();

  if (text.includes("forest") || text.includes("wood") || text.includes("grove")) return "forest";
  if (text.includes("tavern") || text.includes("inn") || text.includes("pub")) return "tavern";
  if (text.includes("storm") || text.includes("rain") || text.includes("thunder")) return "storm";
  if (text.includes("dungeon") || text.includes("cave") || text.includes("crypt")) return "dungeon";
  if (text.includes("market") || text.includes("city") || text.includes("bazaar")) return "city_market";

  return "none";
}

/**
 * Start playing ambient audio for the given scene (by inferred type).
 * No-op if type is "none" or URL is empty.
 */
export function playSceneAmbient(scene: Scene, options?: { volume?: number }): void {
  stopSceneAmbient();
  const type = inferAmbientType(scene);
  if (type === "none") return;

  const url = AMBIENT_URLS[type];
  if (!url) return;

  const audio = new Audio(url);
  audio.loop = true;
  audio.volume = Math.min(1, Math.max(0, options?.volume ?? 0.3));
  audio.play().catch(() => {});
  activeAudio = audio;
}

/**
 * Stop any currently playing scene ambient audio.
 */
export function stopSceneAmbient(): void {
  if (activeAudio) {
    activeAudio.pause();
    activeAudio.currentTime = 0;
    activeAudio = null;
  }
}

export function getActiveSceneAmbient(): HTMLAudioElement | null {
  return activeAudio;
}
