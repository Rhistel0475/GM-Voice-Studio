import React from "react";
import { ScrollText } from "lucide-react";

/**
 * Single quick-access tool card for the GM control panel. Supports image or icon fallback.
 */
export default function QuickToolCard({ tool, onClick }) {
  const hasImg = tool.img && !String(tool.img).endsWith("/undefined");
  const Icon = tool.icon ?? (tool.id === "narration" ? ScrollText : null);

  return (
    <button
      type="button"
      className="quick-tool-card w-full min-h-[88px] p-0 relative overflow-hidden rounded-lg flex flex-col items-center justify-center transition-all duration-200 hover:shadow-[0_0_16px_rgba(202,167,75,0.35)] hover:-translate-y-0.5"
      onClick={() => onClick?.(tool)}
      title={tool.name}
    >
      {hasImg ? (
        <img
          src={tool.img}
          alt=""
          className="absolute inset-0 w-full h-full object-cover block"
        />
      ) : Icon ? (
        <span className="flex-1 flex items-center justify-center text-[var(--gold)]/90 p-3" aria-hidden>
          <Icon size={32} strokeWidth={1.5} />
        </span>
      ) : null}
      <span className="absolute bottom-0 left-0 right-0 bg-black/70 text-[#e7c27a] font-heading text-[10px] text-center py-1.5 px-1 tracking-wide leading-tight">
        {tool.name}
      </span>
    </button>
  );
}
