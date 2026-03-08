import React from "react";

/**
 * Center panel: contains SessionLog, NarrationComposer, AudioPlaybackCard (via children from App).
 * Section header "Live Session" is rendered by LiveBoardPage.
 */
export default function SessionStream({ children }) {
  return (
    <div className="h-full min-h-0 flex flex-col rounded-b-lg overflow-hidden">
      {children}
    </div>
  );
}
