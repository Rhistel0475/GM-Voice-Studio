import React from "react";

export default function PanelShell({ title, children, className = "" }) {
  return (
    <section className={`panel-ornate ${className}`}>
      <div className="panel-head">
        <div className="plaque">{title}</div>
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}
