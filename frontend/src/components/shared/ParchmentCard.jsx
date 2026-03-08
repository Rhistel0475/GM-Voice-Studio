import React from "react";

/**
 * Base container with optional title, header actions, and content.
 * Uses parchment/wood styles from styles.css.
 */
export default function ParchmentCard({ title, headerAction, children, className = "" }) {
  return (
    <div className={`parchment rounded overflow-hidden ${className}`.trim()}>
      {(title || headerAction) && (
        <div className="panel-head flex items-center justify-between gap-2">
          {title && <span className="font-heading text-[var(--text-1)]">{title}</span>}
          {headerAction}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </div>
  );
}
