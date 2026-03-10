import React from "react";

/**
 * Page header for workspace views: icon, title, optional subtitle.
 * icon can be a string (emoji) or a Lucide component.
 */
export default function WorkspaceHeader({ icon, title, subtitle, action, className = "" }) {
  const iconEl =
    typeof icon === "string" ? (
      <span className="text-xl leading-none" aria-hidden>{icon}</span>
    ) : icon
      ? (() => {
          const Icon = icon;
          return <Icon size={22} className="text-[var(--gold)] shrink-0" />;
        })()
      : null;

  return (
    <header className={`flex flex-wrap items-center justify-between gap-4 mb-4 ${className}`.trim()}>
      <div className="flex items-center gap-3">
        {iconEl}
        <div>
          <h1 className="font-heading text-[var(--gold)] text-xl md:text-2xl leading-tight tracking-wide">
            {title}
          </h1>
          {subtitle && (
            <p className="text-sm text-[var(--text-2)] mt-1">{subtitle}</p>
          )}
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </header>
  );
}
