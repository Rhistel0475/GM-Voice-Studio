/**
 * Scene trigger normalization for the LiveBoard Scene Control panel.
 * Mirrors the backend fallback behavior closely enough for button rendering.
 */

const TYPE_ALIASES = {
  narration: "narration",
  narrate: "narration",
  read_aloud: "narration",
  readaloud: "narration",
  dialogue: "dialogue",
  speak: "dialogue",
  npc_dialogue: "dialogue",
  npcdialogue: "dialogue",
  ai_action: "ai_action",
  ai: "ai_action",
  action: "ai_action",
};

function normalizeTriggerType(value) {
  const raw = String(value || "").trim().toLowerCase().replace(/-/g, "_").replace(/\s+/g, "_");
  return TYPE_ALIASES[raw] || raw || "ai_action";
}

function sceneText(scene) {
  return (
    String(scene?.read_aloud || "").trim()
    || String(scene?.notes || "").trim()
    || String(scene?.summary || "").trim()
    || String(scene?.title || "").trim()
  );
}

function coerceAction(action) {
  if (action && typeof action === "object") return { ...action };
  if (typeof action === "string" && action.trim()) return { prompt: action.trim() };
  return null;
}

function defaultDialoguePrompt(scene, npcName, greeting = false) {
  const context = sceneText(scene) || `the scene titled ${String(scene?.title || "Unknown Scene").trim()}`;
  if (greeting) {
    return `Offer a brief in-character greeting or opening reaction as ${npcName} for this scene: ${context}`;
  }
  return `Respond in character as ${npcName} to the current scene situation. Keep it short and table-ready. Context: ${context}`;
}

function dedupeTriggers(triggers) {
  const seen = new Set();
  return triggers.filter((trigger) => {
    const name = String(trigger?.name || "").trim();
    if (!name) return false;
    const key = name.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function buildFallbackTriggers(scene) {
  const triggers = [];
  const readAloud = String(scene?.read_aloud || "").trim();
  const notes = String(scene?.notes || "").trim();

  if (readAloud) {
    triggers.push({ name: "Narrate Scene", type: "narration", text: readAloud });
  } else if (notes) {
    triggers.push({ name: "Narrate Scene", type: "narration", text: notes });
  }

  if (notes && notes !== readAloud) {
    triggers.push({ name: "Reveal Lore", type: "narration", text: notes });
  }

  const primaryNpc = Array.isArray(scene?.npcs) ? scene.npcs.find(Boolean) : "";
  if (primaryNpc) {
    triggers.push({
      name: `Speak as ${primaryNpc}`,
      type: "dialogue",
      action: {
        kind: "generate_dialogue",
        npc_name: primaryNpc,
        prompt: defaultDialoguePrompt(scene, primaryNpc, true),
      },
    });
    triggers.push({
      name: "Generate Dialogue",
      type: "dialogue",
      action: {
        kind: "generate_dialogue",
        npc_name: primaryNpc,
        prompt: defaultDialoguePrompt(scene, primaryNpc, false),
      },
    });
  }

  return dedupeTriggers(triggers);
}

export function normalizeSceneTriggers(scene) {
  const explicit = Array.isArray(scene?.triggers) ? scene.triggers : [];
  const normalized = explicit
    .map((trigger) => {
      if (!trigger || typeof trigger !== "object") return null;

      let name = String(trigger.name || "").trim();
      const type = normalizeTriggerType(trigger.type);
      const text = typeof trigger.text === "string" ? trigger.text.trim() : "";
      const action = coerceAction(trigger.action);

      if (!name) {
        if (type === "narration") name = "Narrate Scene";
        else if (type === "dialogue") name = "Speak Dialogue";
        else name = "Run Scene Action";
      }

      if (!text && !action) {
        if (type === "narration") {
          return { name, type, text: sceneText(scene), action: null };
        }
        if (type === "dialogue") {
          const primaryNpc = Array.isArray(scene?.npcs) ? scene.npcs.find(Boolean) : "";
          if (primaryNpc) {
            return {
              name,
              type,
              text: "",
              action: {
                kind: "generate_dialogue",
                npc_name: primaryNpc,
                prompt: defaultDialoguePrompt(scene, primaryNpc, true),
              },
            };
          }
        }
        return null;
      }

      return { name, type, text, action };
    })
    .filter(Boolean);

  return normalized.length ? dedupeTriggers(normalized) : buildFallbackTriggers(scene);
}
