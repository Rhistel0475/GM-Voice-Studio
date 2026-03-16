import { normalizeSceneTriggers } from "../../lib/sceneTriggers";

function formatTriggerType(triggerType) {
  const value = String(triggerType || "").trim().replace(/[_-]+/g, " ");
  return value || "trigger";
}

export default function SceneControlPanel({
  scene,
  onTrigger,
  busyTriggerName = "",
  triggerError = "",
}) {
  const triggers = normalizeSceneTriggers(scene);

  return (
    <section className="panel-ornate min-h-0 flex flex-col rounded-lg overflow-hidden">
      <div className="panel-head">
        <div className="plaque">Scene Control</div>
      </div>
      <div className="panel-body min-h-0">
        {triggers.length > 0 ? (
          <div className="scene-trigger-list">
            {triggers.map((trigger) => {
              const isBusy = busyTriggerName === trigger.name;
              return (
                <button
                  key={trigger.name}
                  type="button"
                  className="scene-trigger-btn"
                  onClick={() => onTrigger?.(trigger)}
                  disabled={isBusy || !onTrigger}
                >
                  <span>{isBusy ? "Running..." : trigger.name}</span>
                  <span className="scene-trigger-type">{formatTriggerType(trigger.type)}</span>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="intake-empty text-xs">
            No scene triggers are configured for the active scene yet.
          </div>
        )}
        {triggerError ? <div className="text-xs text-red-400 mt-2">{triggerError}</div> : null}
      </div>
    </section>
  );
}
