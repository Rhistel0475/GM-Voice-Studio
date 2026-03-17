const BACKEND_URL = import.meta.env.DEV ? "http://localhost:7862" : "";

export const QUICK_TOOLS = [
  { id: "dice", name: "Roll Dice", img: `${BACKEND_URL}/static/img/Dices.png` },
  { id: "npc", name: "NPC Generator", img: `${BACKEND_URL}/static/img/Maps.png` },
  { id: "loot", name: "Loot Table", img: `${BACKEND_URL}/static/img/Loottable.png` },
  { id: "spells", name: "Spell Reference", img: `${BACKEND_URL}/static/img/Spellbook.png` },
  { id: "monster", name: "Monster Lookup", img: `${BACKEND_URL}/static/img/Swords.png` },
  { id: "narration", name: "Quick Narration", img: null },
];

export const CODEX_TABS = [
  { key: "npcs", label: "NPCs" },
  { key: "scenes", label: "Scenes" },
  { key: "locations", label: "Locations" },
  { key: "quests", label: "Quests" },
  { key: "factions", label: "Factions" },
];
