import React from "react";
import QuickToolCard from "./QuickToolCard";

export default function QuickToolGrid({ tools, onToolClick }) {
  return (
    <div className="grid grid-cols-3 gap-2 h-full">
      {tools.map((tool) => (
        <QuickToolCard key={tool.id} tool={tool} onClick={onToolClick} />
      ))}
    </div>
  );
}
