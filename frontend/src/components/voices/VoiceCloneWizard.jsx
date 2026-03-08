import React from "react";
import { FantasyButton } from "../shared";

const STEPS = [
  { key: 1, label: "Upload" },
  { key: 2, label: "Train" },
  { key: 3, label: "Save" },
];

export default function VoiceCloneWizard({
  step,
  cloneFile,
  onFileChange,
  cloneName,
  onNameChange,
  onTrain,
  isCloning,
  cloneStatus,
  cloneProgress,
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 text-sm text-[var(--text-2)]">
        {STEPS.map((s, i) => (
          <React.Fragment key={s.key}>
            <span className={step >= s.key ? "text-[var(--gold)]" : ""}>{s.key}. {s.label}</span>
            {i < STEPS.length - 1 && <span>→</span>}
          </React.Fragment>
        ))}
      </div>
      {step === 1 && (
        <>
          <label className="field-wrap">
            <span>Audio sample (WAV/MP3, 3–120 s)</span>
            <input
              type="file"
              accept="audio/*"
              className="chat-input"
              onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            />
          </label>
          <label className="field-wrap">
            <span>Voice name (optional)</span>
            <input
              type="text"
              placeholder="Display name"
              className="chat-input"
              value={cloneName}
              onChange={(e) => onNameChange(e.target.value)}
            />
          </label>
          <FantasyButton
            variant="primary"
            className="w-full"
            onClick={onTrain}
            disabled={!cloneFile || isCloning}
          >
            {isCloning ? "Training…" : "Train voice"}
          </FantasyButton>
        </>
      )}
      {(step === 2 || isCloning) && (
        <>
          <div className="w-full h-2 bg-[#1a1008] rounded overflow-hidden">
            <div
              className="h-full bg-[var(--gold)] transition-all duration-300"
              style={{ width: `${cloneProgress}%` }}
            />
          </div>
          <p className="text-sm text-[var(--text-2)]">{cloneStatus}</p>
        </>
      )}
      {step === 3 && (
        <p className="text-sm text-[var(--gold)]">Voice saved. It appears in the library above.</p>
      )}
    </div>
  );
}
