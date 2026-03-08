import React from "react";

export default function QuickToolCard({ tool, onClick }) {
  return (
    <button
      type="button"
      className="quick-tile w-full h-full p-0 relative overflow-hidden"
      onClick={() => onClick?.(tool)}
    >
      <img
        src={tool.img}
        alt={tool.name}
        className="w-full h-full object-cover block"
      />
      <span className="absolute bottom-0 left-0 right-0 bg-black/55 text-[#e7c27a] font-heading text-[9px] text-center py-0.5 px-0.5 letter-spacing:0.04em leading-tight">
        {tool.name}
      </span>
    </button>
  );
}
