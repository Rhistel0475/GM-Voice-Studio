const BACKEND_URL = import.meta.env.DEV ? "http://localhost:7862" : "";

export const QUICK_TOOLS = [
  { id: "dice", name: "Roll Dice", img: `${BACKEND_URL}/static/img/Dices.png` },
  { id: "spells", name: "Grimoire", img: `${BACKEND_URL}/static/img/Spellbook.png` },
  { id: "map", name: "World Map", img: `${BACKEND_URL}/static/img/Maps.png` },
  { id: "loot", name: "Loot Table", img: `${BACKEND_URL}/static/img/Loottable.png` },
  { id: "combat", name: "Encounter", img: `${BACKEND_URL}/static/img/Swords.png` },
  { id: "settings", name: "Settings", img: `${BACKEND_URL}/static/img/Settings.png` },
];

export const CODEX_TABS = [
  { key: "documents", label: "Documents" },
  { key: "npcs", label: "NPCs" },
  { key: "locations", label: "Locations" },
  { key: "rules", label: "Rules" },
];
