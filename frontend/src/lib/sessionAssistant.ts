import { normalizeSceneTriggers } from "./sceneTriggers";

export interface SessionAssistantNpc {
  name: string;
  tone: string;
  description: string;
  raw: Record<string, any>;
}

export interface SessionAssistantQuest {
  name: string;
  summary: string;
  raw: Record<string, any>;
}

export interface SessionAssistantLogEntry {
  id: string;
  role: string;
  text: string;
  meta?: string;
}

export interface SessionAssistantContext {
  npcsInScene: SessionAssistantNpc[];
  currentLocation: string;
  activeQuests: SessionAssistantQuest[];
  recentEvents: SessionAssistantLogEntry[];
}

export interface SessionAssistantSuggestionAction {
  kind:
    | "npc_dialogue"
    | "narrate_text"
    | "brain_query"
    | "start_combat"
    | "expand_scene_description"
    | "add_scene_twist"
    | "narrate_scene"
    | "npc_whisper";
  npcName?: string;
  text?: string;
  query?: string;
}

export interface SessionAssistantSuggestion {
  id: string;
  title: string;
  description: string;
  type: string;
  npcName?: string;
  source: "assistant" | "context";
  runAction: SessionAssistantSuggestionAction;
  narrateText: string;
}

function text(value: unknown): string {
  return String(value || "").trim();
}

function uniqueStrings(values: unknown[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const next = text(value);
    const key = next.toLowerCase();
    if (!next || seen.has(key)) continue;
    seen.add(key);
    out.push(next);
  }
  return out;
}

function toneFromNpc(npc: Record<string, any>): string {
  const tags = Array.isArray(npc?.tags) ? npc.tags.map((value: unknown) => text(value)).filter(Boolean) : [];
  return (
    text(npc?.personality)
    || text(npc?.role)
    || uniqueStrings(tags).slice(0, 2).join(" · ")
    || "Unclear tone"
  );
}

export function deriveNpcTone(npc: Record<string, any>): string {
  return toneFromNpc(npc);
}

function resolveSceneNpcNames(scene: Record<string, any> | null | undefined): string[] {
  const direct = Array.isArray(scene?.npcs) ? scene.npcs : [];
  const related = Array.isArray(scene?.relatedNpcNames) ? scene.relatedNpcNames : [];
  return uniqueStrings([...direct, ...related]);
}

function resolveSceneQuestNames(scene: Record<string, any> | null | undefined): string[] {
  return uniqueStrings(Array.isArray(scene?.relatedQuestNames) ? scene.relatedQuestNames : []);
}

function relatedQuestMatchesScene(
  quest: Record<string, any>,
  sceneNpcNames: string[],
  sceneLocation: string,
  sceneQuestNames: string[],
  sceneTitle: string,
): boolean {
  const questName = text(quest?.name || quest?.title || quest?.id);
  const relatedNpcs = uniqueStrings(Array.isArray(quest?.related_npcs) ? quest.related_npcs : []);
  const relatedLocations = uniqueStrings(Array.isArray(quest?.related_locations) ? quest.related_locations : []);
  const relatedScenes = uniqueStrings(Array.isArray(quest?.related_scenes) ? quest.related_scenes : []);
  return (
    (questName && sceneQuestNames.some((value) => value.toLowerCase() === questName.toLowerCase()))
    || relatedNpcs.some((value) => sceneNpcNames.some((name) => name.toLowerCase() === value.toLowerCase()))
    || relatedLocations.some((value) => sceneLocation && value.toLowerCase() === sceneLocation.toLowerCase())
    || relatedScenes.some((value) => sceneTitle && value.toLowerCase() === sceneTitle.toLowerCase())
  );
}

export function buildSessionAssistantContext({
  campaign,
  scene,
  actionLog,
}: {
  campaign: Record<string, any> | null | undefined;
  scene: Record<string, any> | null | undefined;
  actionLog: Array<Record<string, any>> | null | undefined;
}): SessionAssistantContext {
  const campaignNpcs = Array.isArray(campaign?.npcs) ? campaign.npcs : [];
  const campaignQuests = Array.isArray(campaign?.quests) ? campaign.quests : [];
  const sceneNpcNames = resolveSceneNpcNames(scene);
  const sceneTitle = text(scene?.title || scene?.name);
  const sceneLocation = text(scene?.location || scene?.locationSummary || scene?.relatedLocationNames?.[0]);

  const npcsInScene = sceneNpcNames.map((name) => {
    const npc = campaignNpcs.find((entry: Record<string, any>) => text(entry?.name).toLowerCase() === name.toLowerCase()) || { name };
    return {
      name,
      tone: toneFromNpc(npc),
      description: text(npc?.description || npc?.summary || npc?.personality || npc?.role),
      raw: npc,
    };
  });

  const sceneQuestNames = resolveSceneQuestNames(scene);
  const activeQuests = campaignQuests
    .filter((quest: Record<string, any>) => relatedQuestMatchesScene(quest, sceneNpcNames, sceneLocation, sceneQuestNames, sceneTitle))
    .map((quest: Record<string, any>) => ({
      name: text(quest?.name || quest?.title || quest?.id),
      summary: text(quest?.objective || quest?.description || quest?.summary || quest?.stakes),
      raw: quest,
    }))
    .filter((quest: SessionAssistantQuest) => Boolean(quest.name));

  const recentEvents = (Array.isArray(actionLog) ? actionLog : [])
    .slice()
    .reverse()
    .filter((entry) => text(entry?.text))
    .slice(0, 8)
    .map((entry, index) => ({
      id: text(entry?.id) || `session-log-${index}`,
      role: text(entry?.role || entry?.type) || "system",
      text: text(entry?.text),
      meta: text(entry?.meta),
    }));

  return {
    npcsInScene,
    currentLocation: sceneLocation,
    activeQuests,
    recentEvents,
  };
}

function looksCombatReady(scene: Record<string, any> | null | undefined): boolean {
  const blob = [
    text(scene?.type),
    text(scene?.title),
    text(scene?.summary),
    text(scene?.notes),
    text(scene?.read_aloud),
  ].join(" ").toLowerCase();
  return /combat|ambush|battle|fight|attack|skirmish|encounter/.test(blob);
}

function aiSuggestionToAction(suggestion: Record<string, any>): SessionAssistantSuggestionAction {
  const suggestionType = text(suggestion?.type).toLowerCase();
  if (suggestionType === "npc_dialogue") {
    return {
      kind: "npc_dialogue",
      npcName: text(suggestion?.npc_name),
      text: text(suggestion?.spoken_text || suggestion?.text),
    };
  }
  if (suggestionType === "narration") {
    return {
      kind: "narrate_text",
      text: text(suggestion?.spoken_text || suggestion?.text),
    };
  }
  return {
    kind: "brain_query",
    query: text(suggestion?.action_prompt || suggestion?.text),
  };
}

function normalizeAiSuggestions(aiSuggestions: Array<Record<string, any>>): SessionAssistantSuggestion[] {
  return aiSuggestions
    .filter((suggestion) => suggestion && typeof suggestion === "object" && text(suggestion?.text))
    .map((suggestion, index) => ({
      id: text(suggestion?.id) || `assistant-${index}-${text(suggestion?.title || suggestion?.type || "suggestion").toLowerCase().replace(/\s+/g, "-")}`,
      title: text(suggestion?.title) || "Suggestion",
      description: text(suggestion?.text),
      type: text(suggestion?.type) || "suggestion",
      npcName: text(suggestion?.npc_name) || undefined,
      source: "assistant" as const,
      runAction: aiSuggestionToAction(suggestion),
      narrateText: text(suggestion?.spoken_text || suggestion?.text),
    }));
}

function buildContextSuggestions(
  scene: Record<string, any> | null | undefined,
  context: SessionAssistantContext,
): SessionAssistantSuggestion[] {
  if (!scene) return [];

  const suggestions: SessionAssistantSuggestion[] = [];
  const sceneId = text(scene?.id || scene?.title || "scene");
  const sceneTitle = text(scene?.title || scene?.name || "this scene");
  const sceneText = text(scene?.read_aloud || scene?.notes || scene?.summary);
  const primaryNpc = context.npcsInScene[0];
  const firstQuest = context.activeQuests[0];
  const triggers = normalizeSceneTriggers(scene);
  const hasNarrationTrigger = triggers.some((trigger: Record<string, any>) => text(trigger?.type).toLowerCase() === "narration");

  if (sceneText || hasNarrationTrigger) {
    suggestions.push({
      id: `context-read-${sceneId}`,
      title: "Read Aloud",
      description: `Deliver the strongest table-ready intro for ${sceneTitle}.`,
      type: "scene",
      source: "context",
      runAction: { kind: "narrate_scene", text: sceneText },
      narrateText: sceneText,
    });
  }

  suggestions.push({
    id: `context-expand-${sceneId}`,
    title: "Expand Description",
    description: "Pull richer sensory detail from the current scene without changing the core setup.",
    type: "scene",
    source: "context",
    runAction: { kind: "expand_scene_description" },
    narrateText: `The details of ${sceneTitle} sharpen as the moment unfolds.`,
  });

  suggestions.push({
    id: `context-twist-${sceneId}`,
    title: "Add Twist",
    description: "Surface one table-ready complication or escalation that fits what just happened.",
    type: "scene",
    source: "context",
    runAction: { kind: "add_scene_twist" },
    narrateText: `A new complication cuts into ${sceneTitle} without warning.`,
  });

  if (primaryNpc?.name) {
    suggestions.push({
      id: `context-whisper-${primaryNpc.name.toLowerCase().replace(/\s+/g, "-")}`,
      title: `Whisper from ${primaryNpc.name}`,
      description: `Prompt a brief aside in ${primaryNpc.name}'s tone for the current moment.`,
      type: "npc",
      npcName: primaryNpc.name,
      source: "context",
      runAction: { kind: "npc_whisper", npcName: primaryNpc.name },
      narrateText: `${primaryNpc.name} draws close with a low, private warning.`,
    });
  }

  if (firstQuest?.name) {
    suggestions.push({
      id: `context-quest-${firstQuest.name.toLowerCase().replace(/\s+/g, "-")}`,
      title: `Advance ${firstQuest.name}`,
      description: `Tie the scene back to the active quest stakes or objective.`,
      type: "quest",
      source: "context",
      runAction: {
        kind: "brain_query",
        query: [
          `Give the GM one concise way to connect the active scene to the quest "${firstQuest.name}".`,
          firstQuest.summary ? `Quest context: ${firstQuest.summary}.` : "",
          context.currentLocation ? `Location: ${context.currentLocation}.` : "",
          primaryNpc?.name ? `NPC in focus: ${primaryNpc.name}.` : "",
          `Return only the table-ready suggestion.`,
        ].filter(Boolean).join(" "),
      },
      narrateText: firstQuest.summary || `The pressure around ${firstQuest.name} grows sharper in this scene.`,
    });
  }

  if (looksCombatReady(scene)) {
    suggestions.push({
      id: `context-combat-${sceneId}`,
      title: "Start Combat",
      description: "Kick off encounter flow, combat ambience, and opening pressure for the active scene.",
      type: "encounter",
      source: "context",
      runAction: { kind: "start_combat" },
      narrateText: `The tension in ${sceneTitle} snaps and the encounter breaks into open conflict.`,
    });
  }

  return suggestions;
}

export function buildSessionAssistantSuggestions({
  aiSuggestions,
  context,
  scene,
}: {
  aiSuggestions: Array<Record<string, any>> | null | undefined;
  context: SessionAssistantContext;
  scene: Record<string, any> | null | undefined;
}): SessionAssistantSuggestion[] {
  const merged: SessionAssistantSuggestion[] = [];
  const seen = new Set<string>();

  const pushUnique = (suggestion: SessionAssistantSuggestion) => {
    const key = `${suggestion.title}|${suggestion.description}`.toLowerCase();
    if (!suggestion.title || !suggestion.description || seen.has(key)) return;
    seen.add(key);
    merged.push(suggestion);
  };

  normalizeAiSuggestions(Array.isArray(aiSuggestions) ? aiSuggestions : []).forEach(pushUnique);
  buildContextSuggestions(scene, context).forEach(pushUnique);

  return merged.slice(0, 5);
}
