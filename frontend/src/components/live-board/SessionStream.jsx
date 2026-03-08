import React from "react";
import SectionHeader from "../layout/SectionHeader";

/**
 * Wraps the live session middle column content with a section header.
 * Pass the full middle column (e.g. MiddleColumn from App) as children.
 */
export default function SessionStream({ children }) {
  return (
    <div className="h-full min-h-0 flex flex-col">
      <SectionHeader title="Live Session" />
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}
