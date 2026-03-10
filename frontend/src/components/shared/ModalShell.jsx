import React from "react";

/**
 * Overlay + panel for modals; reuses panel style.
 */
export default function ModalShell({ title, onClose, children, className = "" }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#0a0603]/85 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? "modal-title" : undefined}
      onClick={onClose || undefined}
    >
      <div
        className={`panel-ornate max-h-[90vh] w-full max-w-lg overflow-hidden rounded-lg ${className}`.trim()}
        onClick={(e) => e.stopPropagation()}
      >
        {(title || onClose) && (
          <div className="panel-head panel-head--row flex items-center justify-between gap-2">
            {title && (
              <h2 id="modal-title" className="font-heading text-[var(--text-1)] text-base">
                {title}
              </h2>
            )}
            {onClose && (
              <button
                type="button"
                className="ml-auto rounded p-1.5 text-[#9c7a3a] hover:bg-[rgba(255,210,122,0.12)] hover:text-[var(--text-1)] transition-colors"
                onClick={onClose}
                aria-label="Close"
              >
                ×
              </button>
            )}
          </div>
        )}
        <div className="panel-body overflow-auto">{children}</div>
      </div>
    </div>
  );
}
