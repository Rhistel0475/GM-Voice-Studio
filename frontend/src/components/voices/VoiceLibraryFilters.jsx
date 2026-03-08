import React from "react";
import { VOICE_SOURCES, VOICE_STATUSES, VOICE_TONES } from "../../types/voice";

const SOURCE_LABELS = { system: "System", cloned: "Cloned", custom: "Custom", all: "All sources" };
const STATUS_LABELS = { ready: "Ready", training: "Training", failed: "Failed", draft: "Draft", all: "All statuses" };
const TONE_LABELS = {
  warm: "Warm", grim: "Grim", noble: "Noble", mysterious: "Mysterious", rough: "Rough", neutral: "Neutral", all: "All tones",
};

/**
 * Filters for voice library: source, status, tone. Dropdowns with All option.
 */
export default function VoiceLibraryFilters({ filterState, onFilterChange }) {
  const setFilter = (patch) => onFilterChange({ ...filterState, ...patch });

  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-heading text-[var(--text-2)] uppercase tracking-wider">
        Filter
      </div>
      <label className="field-wrap">
        <span>Source</span>
        <select
          className="chat-input w-full text-sm"
          value={filterState.source || "all"}
          onChange={(e) => setFilter({ source: e.target.value })}
        >
          <option value="all">All sources</option>
          {VOICE_SOURCES.map((s) => (
            <option key={s} value={s}>
              {SOURCE_LABELS[s] || s}
            </option>
          ))}
        </select>
      </label>
      <label className="field-wrap">
        <span>Status</span>
        <select
          className="chat-input w-full text-sm"
          value={filterState.status || "all"}
          onChange={(e) => setFilter({ status: e.target.value })}
        >
          <option value="all">All statuses</option>
          {VOICE_STATUSES.map((s) => (
            <option key={s} value={s}>
              {STATUS_LABELS[s] || s}
            </option>
          ))}
        </select>
      </label>
      <label className="field-wrap">
        <span>Tone</span>
        <select
          className="chat-input w-full text-sm"
          value={filterState.tone || "all"}
          onChange={(e) => setFilter({ tone: e.target.value })}
        >
          <option value="all">All tones</option>
          {VOICE_TONES.map((t) => (
            <option key={t} value={t}>
              {TONE_LABELS[t] || t}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
