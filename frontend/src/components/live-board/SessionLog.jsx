import React, { useRef, useEffect } from "react";
import SessionLogEntry from "./SessionLogEntry";
import { EmptyState } from "../shared";

export default function SessionLog({ actionLog = [], liveTranscript = "" }) {
  const logRef = useRef(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [actionLog, liveTranscript]);

  return (
    <div
      ref={logRef}
      className="flex-1 min-h-[180px] overflow-y-auto border border-[#4f341f] bg-[#120a04] rounded-b p-3 space-y-2"
    >
      {actionLog.length > 0 ? (
        actionLog.map((entry) => <SessionLogEntry key={entry.id} entry={entry} />)
      ) : (
        <EmptyState message="No live entries yet. Ask a rules/lore question below." />
      )}
      {liveTranscript && (
        <div className="rounded border border-[#5d472a] bg-[#1a1209] px-2 py-1 text-xs text-[#d7b77d]">
          <div className="mb-0.5 flex items-center justify-between">
            <span className="font-heading text-[10px] tracking-wide uppercase">Listening...</span>
            <span className="text-[10px] opacity-80">STT partial</span>
          </div>
          <div className="whitespace-pre-wrap">{liveTranscript}</div>
        </div>
      )}
    </div>
  );
}
