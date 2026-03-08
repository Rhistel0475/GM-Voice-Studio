/**
 * Voice Studio types. JSDoc for IDE support; repo uses .js.
 */

/**
 * @typedef {"system" | "cloned" | "custom"} VoiceSource
 */

/**
 * @typedef {"ready" | "training" | "failed" | "draft"} VoiceStatus
 */

/**
 * @typedef {"warm" | "grim" | "noble" | "mysterious" | "rough" | "neutral"} VoiceTone
 */

/**
 * @typedef {Object} VoiceProfile
 * @property {string} id
 * @property {string} name
 * @property {VoiceSource} source
 * @property {VoiceStatus} status
 * @property {string} [accent]
 * @property {VoiceTone} [tone]
 * @property {string[]} tags
 * @property {string} [sampleUrl]
 * @property {string[]} [assignedNPCs]
 * @property {string} [description]
 * @property {string} [updatedAt]
 * @property {string} [voice_id]
 */

/**
 * @typedef {Object} GeneratedAudio
 * @property {string} id
 * @property {string} title
 * @property {string} voiceId
 * @property {string} [duration]
 * @property {string} [createdAt]
 * @property {string} [campaign]
 * @property {"narration" | "npc-dialogue" | "scene-readout"} type
 * @property {string} [audioUrl]
 */

/**
 * @typedef {Object} VoiceCloneJob
 * @property {string} id
 * @property {"queued" | "processing" | "ready" | "failed"} status
 * @property {number} progress
 * @property {string} [stepLabel]
 */

/**
 * @typedef {Object} VoiceFilterState
 * @property {string} query
 * @property {VoiceSource | "all"} source
 * @property {VoiceStatus | "all"} status
 * @property {VoiceTone | "all"} [tone]
 */

/** @type {VoiceSource[]} */
export const VOICE_SOURCES = ["system", "cloned", "custom"];

/** @type {VoiceStatus[]} */
export const VOICE_STATUSES = ["ready", "training", "failed", "draft"];

/** @type {VoiceTone[]} */
export const VOICE_TONES = ["warm", "grim", "noble", "mysterious", "rough", "neutral"];

/**
 * @returns {VoiceFilterState}
 */
export function defaultVoiceFilterState() {
  return {
    query: "",
    source: "all",
    status: "all",
    tone: "all",
  };
}
