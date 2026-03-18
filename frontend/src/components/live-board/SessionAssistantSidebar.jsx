import SessionAssistantPanel from "./SessionAssistantPanel";
import CodexSidePanel from "./CodexSidePanel";

export default function SessionAssistantSidebar({
  campaignData,
  authFetch,
  onInsertIntoNarration,
  onNarrateAnswer,
  onSpeakNpc,
  assistantSupported,
  assistantListening,
  assistantAnalyzing,
  assistantError,
  assistantPartialTranscript,
  assistantSuggestions,
  assistantContext,
  actionLog,
  assistantActionBusyId,
  onStartAssistantListening,
  onStopAssistantListening,
  onAnalyzeAssistant,
  onRunAssistantSuggestion,
  onNarrateAssistantSuggestion,
  onIgnoreAssistantSuggestion,
}) {
  return (
    <div className="h-full min-h-0 flex flex-col gap-3">
      <SessionAssistantPanel
        supported={assistantSupported}
        listening={assistantListening}
        analyzing={assistantAnalyzing}
        error={assistantError}
        partialTranscript={assistantPartialTranscript}
        suggestions={assistantSuggestions}
        context={assistantContext}
        sessionLog={actionLog}
        actionBusyId={assistantActionBusyId}
        onStartListening={onStartAssistantListening}
        onStopListening={onStopAssistantListening}
        onAnalyzeNow={onAnalyzeAssistant}
        onRunSuggestion={onRunAssistantSuggestion}
        onNarrateSuggestion={onNarrateAssistantSuggestion}
        onIgnoreSuggestion={onIgnoreAssistantSuggestion}
      />
      <div className="min-h-[280px] max-h-[42%] flex-1 overflow-hidden">
        <CodexSidePanel
          campaignData={campaignData}
          onInsertIntoNarration={onInsertIntoNarration}
          onSpeakNpc={onSpeakNpc}
          authFetch={authFetch}
          onNarrateAnswer={onNarrateAnswer}
          hideBrain
          panelTitle="Campaign Knowledge"
        />
      </div>
    </div>
  );
}
