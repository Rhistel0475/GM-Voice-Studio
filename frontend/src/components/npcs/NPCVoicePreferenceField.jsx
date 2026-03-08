import React from "react";

/**
 * Preferred voice dropdown for the NPC (form field).
 */
export default function NPCVoicePreferenceField({
  voices = [],
  selectedVoiceId,
  onVoiceChange,
}) {
  return (
    <label className="field-wrap">
      <span>Preferred voice</span>
      <select
        className="chat-input w-full"
        value={selectedVoiceId || ""}
        onChange={(e) => onVoiceChange(e.target.value || undefined)}
      >
        <option value="">— None —</option>
        {voices.map((v) => (
          <option key={v.voice_id} value={v.voice_id}>
            {v.name?.trim() || v.voice_id}
          </option>
        ))}
      </select>
    </label>
  );
}
