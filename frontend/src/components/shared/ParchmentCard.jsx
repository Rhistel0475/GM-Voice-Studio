import React from "react";

/**
 * Base container with optional title, header actions, content, and footer.
 * Uses parchment/wood styles from styles.css.
 */
export default function ParchmentCard({ title, headerAction, children, footer, className = "" }) {
  return (
    <div className={`parchment rounded overflow-hidden ${className}`.trim()}>
      {(title || headerAction) && (
        <div className="panel-head panel-head--row flex items-center justify-between gap-2">
          {title && (
            <span className="font-heading text-base tracking-wide">
              {title}
            </span>
          )}
          {headerAction}
        </div>
      )}
      <div className="panel-body">{children}</div>
      {footer && (
        <div className="panel-head panel-head--row border-t border-[#5c3e23] flex items-center justify-between gap-2">
          {footer}
        </div>
      )}
    </div>
  );
}
