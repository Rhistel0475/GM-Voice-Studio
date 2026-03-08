import React from "react";
import { TabGroup } from "../shared";
import { CODEX_TABS } from "./constants";

export default function CodexTabs({ selectedKey, onChange }) {
  return <TabGroup tabs={CODEX_TABS} selectedKey={selectedKey} onChange={onChange} className="mb-2" />;
}
