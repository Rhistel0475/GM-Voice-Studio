import React from "react";

/**
 * Tab strip with selected key and onChange(key). Renders tabs from tabs array of { key, label }.
 */
export default function TabGroup({ tabs, selectedKey, onChange, className = "" }) {
  return (
    <div className={`tab-strip ${className}`.trim()}>
      {tabs.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={selectedKey === key ? "tab-active" : ""}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
