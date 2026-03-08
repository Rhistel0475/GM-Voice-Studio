import React from "react";
import { ChevronLeft } from "lucide-react";
import { FantasyButton } from "../shared";

/**
 * Step 4: Name, tags, and save. Final step of clone wizard.
 */
export default function VoiceSaveForm({
  name,
  onNameChange,
  tags = [],
  onTagsChange,
  onSave,
  onBack,
  saving,
  saved,
  error,
}) {
  const tagStr = Array.isArray(tags) ? tags.join(", ") : "";
  const setTagStr = (s) => {
    const next = s.split(/[\s,]+/).filter(Boolean);
    onTagsChange?.(next);
  };

  if (saved) {
    return (
      <div className="rounded border border-[#5f7d63] bg-[rgba(94,124,99,0.15)] p-3 text-[#9ec24f] text-sm font-heading">
        Voice saved. It appears in the library.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <label className="field-wrap">
        <span>Voice name</span>
        <input
          type="text"
          placeholder="Display name"
          className="chat-input w-full"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          disabled={saving}
        />
      </label>
      <label className="field-wrap">
        <span>Tags (comma-separated)</span>
        <input
          type="text"
          placeholder="e.g. tavern, friendly, narrator"
          className="chat-input w-full"
          value={tagStr}
          onChange={(e) => setTagStr(e.target.value)}
          disabled={saving}
        />
      </label>
      {error && <p className="text-sm text-red-400">{error}</p>}
      <div className="flex gap-2">
        {onBack && (
          <FantasyButton variant="ghost" className="flex-1" onClick={onBack} disabled={saving}>
            <ChevronLeft size={14} className="inline mr-1" /> Back
          </FantasyButton>
        )}
        <FantasyButton
          variant="primary"
          className={onBack ? "flex-1" : "w-full"}
          onClick={onSave}
          disabled={!name?.trim() || saving}
        >
          {saving ? "Saving…" : "Save voice"}
        </FantasyButton>
      </div>
    </div>
  );
}
