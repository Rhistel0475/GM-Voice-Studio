import { Fragment } from "react";
import VoiceSampleUpload from "./VoiceSampleUpload";
import VoiceTrainingStatus from "./VoiceTrainingStatus";
import VoicePreviewStep from "./VoicePreviewStep";
import VoiceSaveForm from "./VoiceSaveForm";

const STEPS = [
  { key: 1, label: "Upload" },
  { key: 2, label: "Train" },
  { key: 3, label: "Preview" },
  { key: 4, label: "Name & tag" },
  { key: 5, label: "Save" },
];

/**
 * Multi-step clone wizard: Upload → Train → Preview → Name/tag → Save.
 * Visual stepper, parchment container, Back on steps 3–4.
 */
export default function VoiceCloneWizard({
  step,
  setStep,
  cloneFile,
  onFileChange,
  cloneName,
  onNameChange,
  cloneTags,
  onTagsChange,
  onTrain,
  isCloning,
  cloneStatus,
  cloneProgress,
  cloneVoiceId,
  cloneVoiceName,
  onPlayPreview,
  isPlayingPreview,
  onSave,
  saving,
  saved,
  saveError,
}) {
  const currentStep = Math.min(Math.max(1, step || 1), 5);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-2 flex-wrap" aria-label="Wizard steps">
        {STEPS.map((s, i) => (
          <Fragment key={s.key}>
            <div className="flex items-center gap-1.5">
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-xs font-heading transition-colors ${
                  currentStep > s.key
                    ? "border-[var(--gold)] bg-[var(--gold)] text-[#1a1008]"
                    : currentStep === s.key
                      ? "border-[var(--gold)] bg-[rgba(202,167,75,0.2)] text-[var(--gold)]"
                      : "border-[#5c3e23] text-[var(--text-2)] opacity-60"
                }`}
              >
                {currentStep > s.key ? "✓" : s.key}
              </span>
              <span
                className={`hidden sm:inline text-xs font-heading ${
                  currentStep >= s.key ? "text-[var(--gold)]" : "text-[var(--text-2)] opacity-60"
                }`}
              >
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div className="h-px w-4 sm:w-6 bg-[#5c3e23] shrink-0 opacity-50" aria-hidden />
            )}
          </Fragment>
        ))}
      </div>

      <div className="parchment rounded border border-[#a17a42] p-3 space-y-3">
        {currentStep === 1 && (
          <VoiceSampleUpload
            file={cloneFile}
            onFileChange={onFileChange}
            name={cloneName}
            onNameChange={onNameChange}
            onNext={() => {
              onTrain?.();
              setStep?.(2);
            }}
            disabled={isCloning}
          />
        )}

        {currentStep === 2 && (
          <VoiceTrainingStatus
            status={cloneStatus}
            progress={cloneProgress}
            stepLabel={cloneStatus}
          />
        )}

        {currentStep === 3 && (
          <VoicePreviewStep
            voiceId={cloneVoiceId}
            voiceName={cloneVoiceName}
            onPlayPreview={onPlayPreview}
            isPlaying={isPlayingPreview}
            onNext={() => setStep?.(4)}
            onBack={() => setStep?.(2)}
            disabled={saving}
          />
        )}

        {currentStep === 4 && (
          <VoiceSaveForm
            name={cloneName}
            onNameChange={onNameChange}
            tags={cloneTags}
            onTagsChange={onTagsChange}
            onSave={onSave}
            onBack={() => setStep?.(3)}
            saving={saving}
            saved={saved}
            error={saveError}
          />
        )}

        {currentStep === 5 && (
          <div className="rounded border border-[#5f7d63] bg-[rgba(94,124,99,0.15)] p-3 text-[#9ec24f] text-sm font-heading">
            Voice saved. It appears in the library. You can start another clone or select it to use.
          </div>
        )}
      </div>
    </div>
  );
}
