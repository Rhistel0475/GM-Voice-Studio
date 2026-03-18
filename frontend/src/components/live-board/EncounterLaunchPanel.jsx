function encounterHint(scene) {
  const summary = String(scene?.summary || scene?.notes || scene?.read_aloud || "").trim();
  if (summary) {
    return summary.length > 110 ? `${summary.slice(0, 107).trimEnd()}...` : summary;
  }
  return "Kick off combat narration, battle ambience, and an opening enemy voice line.";
}

export default function EncounterLaunchPanel({
  scene,
  onLaunch,
  isLaunching = false,
  error = "",
}) {
  const sceneTitle = String(scene?.title || scene?.name || "Encounter").trim() || "Encounter";
  const disabled = !scene || !onLaunch || isLaunching;

  return (
    <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
      <div className="panel-head">
        <div className="plaque">Encounter</div>
      </div>
      <div className="panel-body space-y-3">
        <div className="text-xs leading-relaxed text-[#c7a76a]">
          {encounterHint(scene)}
        </div>
        <button
          type="button"
          className="w-full rounded-md border border-[#8a5b34] bg-[#2a170b] px-3 py-3 text-left transition hover:border-[#d4af37] hover:bg-[#341c0e] disabled:cursor-not-allowed disabled:opacity-70"
          onClick={() => onLaunch?.()}
          disabled={disabled}
        >
          <div className="font-heading text-sm uppercase tracking-[0.18em] text-[#e8c787]">
            {isLaunching ? "Starting Combat..." : "Start Combat"}
          </div>
          <div className="mt-1 text-[11px] uppercase tracking-[0.16em] text-[#9b7440]">
            {sceneTitle}
          </div>
        </button>
        {error ? <div className="text-xs text-red-400">{error}</div> : null}
      </div>
    </section>
  );
}
