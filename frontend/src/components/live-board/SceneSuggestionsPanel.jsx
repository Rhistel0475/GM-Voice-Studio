import { Compass } from "lucide-react";

function sceneSummary(scene) {
  const description = String(
    scene?.description || scene?.summary || scene?.read_aloud || scene?.notes || ""
  ).trim();
  if (!description) {
    const parts = [scene?.location, scene?.type].filter(Boolean);
    return parts.length ? parts.join(" • ") : "No scene summary available yet.";
  }
  return description.length > 120 ? `${description.slice(0, 117).trimEnd()}...` : description;
}

export default function SceneSuggestionsPanel({
  suggestions = [],
  isLoading = false,
  error = "",
  onActivate,
  busySceneId = "",
}) {
  return (
    <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
      <div className="panel-head">
        <div className="plaque flex items-center gap-2">
          <Compass size={14} />
          <span>Scene Suggestions</span>
        </div>
      </div>
      <div className="panel-body min-h-0 space-y-2">
        {isLoading ? (
          <div className="intake-empty text-xs">Reading the campaign graph for the next likely scene...</div>
        ) : suggestions.length > 0 ? (
          <div className="space-y-2">
            {suggestions.map((scene) => {
              const sceneId = String(scene?.id || scene?.title || "");
              const isBusy = busySceneId === sceneId;
              return (
                <button
                  key={sceneId}
                  type="button"
                  className="w-full rounded-md border border-[#734f2c] bg-[#211508]/80 px-3 py-3 text-left transition hover:border-[#d4af37] hover:bg-[#2a1a0d] disabled:cursor-not-allowed disabled:opacity-70"
                  onClick={() => onActivate?.(scene)}
                  disabled={!onActivate || isBusy}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-heading text-sm uppercase tracking-[0.18em] text-[#e8c787]">
                      {isBusy ? "Activating..." : `Activate ${scene?.title || scene?.name || "Scene"}`}
                    </span>
                    {scene?.suggestion_reason ? (
                      <span className="text-[10px] uppercase tracking-[0.18em] text-[#9b7440]">
                        {scene.suggestion_reason}
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-1 text-xs leading-relaxed text-[#c7a76a]">
                    {sceneSummary(scene)}
                  </div>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="intake-empty text-xs">
            No next-scene suggestions yet. Add connected scenes or give the party a clearer direction.
          </div>
        )}
        {error ? <div className="text-xs text-red-400">{error}</div> : null}
      </div>
    </section>
  );
}
