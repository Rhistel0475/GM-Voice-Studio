import type {
  Campaign,
  Session,
  Scene,
  Npc,
  Voice,
  CodexEntry,
  ActionLogEvent,
  NarrationClip,
} from "../types";

/** Seed data for the campaign context store. One campaign, one session, linked entities. */
export interface SeedCampaignContext {
  campaigns: Campaign[];
  sessions: Session[];
  scenes: Scene[];
  npcs: Npc[];
  voices: Voice[];
  codexEntries: CodexEntry[];
  actionLog: ActionLogEvent[];
  narrationClips: NarrationClip[];
}

const CAMPAIGN_ID = "camp_stolen_land";
const SESSION_ID = "session_1";
const SCENE_ARRIVAL = "scene_arrival_oleg";
const SCENE_AMBUSH = "scene_ambush_bandits";
const SCENE_THORN = "scene_thorn_river";

const NPC_OLEG = "npc_oleg";
const NPC_SVETLANA = "npc_svetlana";
const NPC_HAPPS = "npc_happs";
const NPC_COURIER = "npc_silver_stag_courier";
const NPC_SCOUT = "npc_nervous_scout";
const NPC_KESSEL = "npc_kessel";
const NPC_BANDIT = "npc_bandit_grunt";

const VOICE_WARM = "voice_warm_narrator";
const VOICE_BANDIT = "voice_rough_bandit";
const VOICE_NOBLE = "voice_noble_trader";
const VOICE_CALM = "voice_calm_guide";
const VOICE_SVETLANA = "voice_svetlana";
const VOICE_SCOUT = "voice_nervous_scout";

const CODEX_OLEG_POST = "codex_oleg_post";
const CODEX_THORN = "codex_thorn_river";
const CODEX_BANDITS = "codex_bandit_pressure";
const CODEX_CHARTER = "codex_lordship_charter";
const CODEX_OUTPOST = "codex_resting_outpost";
const CODEX_RUMORS = "codex_rumors_road";
const CODEX_GREENBELT = "codex_greenbelt";
const CODEX_STAG = "codex_silver_stag";

const LOG_IDS = [
  "log_1",
  "log_2",
  "log_3",
  "log_4",
  "log_5",
  "log_6",
  "log_7",
  "log_8",
  "log_9",
  "log_10",
];
const CLIP_IDS = ["clip_1", "clip_2", "clip_3", "clip_4"];

export const seedCampaignContext: SeedCampaignContext = {
  campaigns: [
    {
      id: CAMPAIGN_ID,
      name: "Stolen Land",
      setting: "River Kingdoms, Greenbelt",
      activeSessionId: SESSION_ID,
    },
  ],
  sessions: [
    {
      id: SESSION_ID,
      campaignId: CAMPAIGN_ID,
      title: "Session 1 — The Road to Oleg's",
      activeSceneId: SCENE_ARRIVAL,
      startedAt: "2025-03-01T14:00:00.000Z",
      status: "active",
    },
  ],
  scenes: [
    {
      id: SCENE_ARRIVAL,
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      title: "Arrival at Oleg's Trading Post",
      summary: "The party reaches the outpost; Oleg and Svetlana greet them.",
      npcIds: [NPC_OLEG, NPC_SVETLANA, NPC_COURIER],
      codexEntryIds: [CODEX_OLEG_POST, CODEX_OUTPOST, CODEX_RUMORS],
      actionLogIds: [LOG_IDS[0], LOG_IDS[1], LOG_IDS[2], LOG_IDS[3]],
      narrationClipIds: [CLIP_IDS[0], CLIP_IDS[1]],
      tags: ["social", "arrival"],
    },
    {
      id: SCENE_AMBUSH,
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      title: "Ambush of the Bandits",
      summary: "Happs Bydon and bandits waylay the party on the trail.",
      npcIds: [NPC_HAPPS, NPC_BANDIT],
      codexEntryIds: [CODEX_BANDITS, CODEX_GREENBELT],
      actionLogIds: [LOG_IDS[4], LOG_IDS[5], LOG_IDS[6], LOG_IDS[7]],
      narrationClipIds: [CLIP_IDS[2]],
      tags: ["combat", "ambush"],
    },
    {
      id: SCENE_THORN,
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      title: "Exploring the Thorn River Bank",
      summary: "The party scouts the river; the nervous mercenary reports in.",
      npcIds: [NPC_SCOUT, NPC_KESSEL],
      codexEntryIds: [CODEX_THORN, CODEX_CHARTER],
      actionLogIds: [LOG_IDS[8], LOG_IDS[9]],
      narrationClipIds: [CLIP_IDS[3]],
      tags: ["exploration", "travel"],
    },
  ],
  npcs: [
    {
      id: NPC_OLEG,
      campaignId: CAMPAIGN_ID,
      name: "Oleg Leveton",
      summary: "Stocky, pragmatic owner of the trading post. Tired of bandit raids.",
      role: "Trader",
      profession: "Merchant",
      voiceId: VOICE_NOBLE,
      tags: ["trader", "oleg's"],
      disposition: "friendly",
    },
    {
      id: NPC_SVETLANA,
      campaignId: CAMPAIGN_ID,
      name: "Svetlana Leveton",
      summary: "Oleg's wife; warm and capable; handles hospitality and ledgers.",
      role: "Host",
      voiceId: VOICE_SVETLANA,
      tags: ["trader", "oleg's"],
      disposition: "friendly",
    },
    {
      id: NPC_HAPPS,
      campaignId: CAMPAIGN_ID,
      name: "Happs Bydon",
      summary: "Brutish bandit leader who preys on travelers. Confident and cruel.",
      role: "Bandit captain",
      voiceId: VOICE_BANDIT,
      tags: ["bandit", "antagonist"],
      disposition: "hostile",
    },
    {
      id: NPC_COURIER,
      campaignId: CAMPAIGN_ID,
      name: "Silver Stag courier",
      summary: "Runner from the Silver Stag company; carries messages and small cargo.",
      role: "Courier",
      voiceId: VOICE_CALM,
      tags: ["courier", "neutral"],
      disposition: "neutral",
    },
    {
      id: NPC_SCOUT,
      campaignId: CAMPAIGN_ID,
      name: "Nervous mercenary scout",
      summary: "Hired scout who's seen too much; jumpy but loyal.",
      role: "Scout",
      voiceId: VOICE_SCOUT,
      tags: ["mercenary", "scout"],
      disposition: "friendly",
    },
    {
      id: NPC_KESSEL,
      campaignId: CAMPAIGN_ID,
      name: "Kessel",
      summary: "Veteran guide familiar with the Thorn River area.",
      role: "Guide",
      voiceId: VOICE_CALM,
      tags: ["guide", "thorn river"],
      disposition: "neutral",
    },
    {
      id: NPC_BANDIT,
      campaignId: CAMPAIGN_ID,
      name: "Bandit grunt",
      summary: "One of Happs's thugs; follows orders.",
      role: "Bandit",
      voiceId: VOICE_BANDIT,
      tags: ["bandit"],
      disposition: "hostile",
    },
  ],
  voices: [
    {
      id: VOICE_WARM,
      campaignId: CAMPAIGN_ID,
      name: "Warm Narrator",
      tone: "warm",
      tags: ["narrator", "gm"],
      assignedNpcIds: [],
      status: "ready",
    },
    {
      id: VOICE_BANDIT,
      campaignId: CAMPAIGN_ID,
      name: "Rough Bandit Captain",
      tone: "rough",
      accent: "gruff",
      tags: ["antagonist", "bandit"],
      assignedNpcIds: [NPC_HAPPS, NPC_BANDIT],
      status: "ready",
    },
    {
      id: VOICE_NOBLE,
      campaignId: CAMPAIGN_ID,
      name: "Noble Trader",
      tone: "measured",
      tags: ["trader", "civilized"],
      assignedNpcIds: [NPC_OLEG],
      status: "ready",
    },
    {
      id: VOICE_CALM,
      campaignId: CAMPAIGN_ID,
      name: "Calm Guide",
      tone: "calm",
      tags: ["guide", "support"],
      assignedNpcIds: [NPC_COURIER, NPC_KESSEL],
      status: "ready",
    },
    {
      id: VOICE_SVETLANA,
      campaignId: CAMPAIGN_ID,
      name: "Svetlana (warm host)",
      tone: "warm",
      tags: ["host", "oleg's"],
      assignedNpcIds: [NPC_SVETLANA],
      status: "ready",
    },
    {
      id: VOICE_SCOUT,
      campaignId: CAMPAIGN_ID,
      name: "Nervous Scout",
      tone: "nervous",
      tags: ["scout", "mercenary"],
      assignedNpcIds: [NPC_SCOUT],
      status: "ready",
    },
  ],
  codexEntries: [
    {
      id: CODEX_OLEG_POST,
      campaignId: CAMPAIGN_ID,
      type: "location",
      title: "Oleg's Trading Post",
      summary: "First outpost in the Greenbelt; run by Oleg and Svetlana Leveton.",
      content: "A fortified trading post on the border. Critical resupply point.",
      relatedNpcIds: [NPC_OLEG, NPC_SVETLANA],
      relatedSceneIds: [SCENE_ARRIVAL],
      tags: ["location", "greenbelt"],
    },
    {
      id: CODEX_THORN,
      campaignId: CAMPAIGN_ID,
      type: "location",
      title: "Thorn River",
      summary: "River running through the Greenbelt; used for travel and trade.",
      relatedSceneIds: [SCENE_THORN],
      tags: ["location", "river"],
    },
    {
      id: CODEX_BANDITS,
      campaignId: CAMPAIGN_ID,
      type: "lore",
      title: "Bandit Pressure in the Greenbelt",
      summary: "Bandits under Happs and others harass travelers and outposts, squeezing every caravan for coin.",
      content: "Oleg reports that the raids have grown bolder each tenday. The bandits know the charter exists, and they are testing how far they can push before a new lord takes the Greenbelt seriously.",
      relatedNpcIds: [NPC_HAPPS],
      relatedSceneIds: [SCENE_AMBUSH],
      tags: ["lore", "bandits"],
    },
    {
      id: CODEX_CHARTER,
      campaignId: CAMPAIGN_ID,
      type: "document",
      title: "Lordship Charter Notes",
      summary: "Notes on claiming and ruling land in the River Kingdoms.",
      tags: ["document", "charter"],
    },
    {
      id: CODEX_OUTPOST,
      campaignId: CAMPAIGN_ID,
      type: "rule",
      title: "Resting at an Outpost",
      summary: "Rules for rest, resupply, and safety at outposts like Oleg's.",
      tags: ["rule", "rest"],
    },
    {
      id: CODEX_RUMORS,
      campaignId: CAMPAIGN_ID,
      type: "lore",
      title: "Rumors from the Road",
      summary: "Travelers' rumors about bandits, lost stag idols, and the Stolen Land's shifting loyalties.",
      content: "Around Oleg's fire the stories always circle back to three things: a stag-masked bandit lord, ruins swallowed by the forest, and Brevic nobles who would rather fund deniable \"adventurers\" than send an army.",
      tags: ["lore", "rumors"],
    },
    {
      id: CODEX_GREENBELT,
      campaignId: CAMPAIGN_ID,
      type: "location",
      title: "The Greenbelt",
      summary: "Wild border region between Brevoy and the River Kingdoms.",
      tags: ["location", "region"],
    },
    {
      id: CODEX_STAG,
      campaignId: CAMPAIGN_ID,
      type: "faction",
      title: "Silver Stag Company",
      summary: "Merchant and courier company operating in the area.",
      relatedNpcIds: [NPC_COURIER],
      tags: ["faction", "trade"],
    },
  ],
  actionLog: [
    {
      id: LOG_IDS[0],
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      type: "narration",
      text: "The trading post rises from the dusty road — palisade walls, a heavy gate, and the smell of smoke and leather.",
      createdAt: "2025-03-01T14:05:00.000Z",
    },
    {
      id: LOG_IDS[1],
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      type: "player",
      text: "We approach the gate and announce ourselves as travelers seeking shelter.",
      createdAt: "2025-03-01T14:06:00.000Z",
    },
    {
      id: LOG_IDS[2],
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      type: "npc",
      text: "Oleg: 'Come in, then. Svetlana'll see you fed. Just don't start trouble.'",
      createdAt: "2025-03-01T14:07:00.000Z",
    },
    {
      id: LOG_IDS[3],
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      type: "system",
      text: "Scene: Arrival at Oleg's Trading Post.",
      createdAt: "2025-03-01T14:04:00.000Z",
    },
    {
      id: LOG_IDS[4],
      sessionId: SESSION_ID,
      sceneId: SCENE_AMBUSH,
      type: "narration",
      text: "The trail narrows between thick brush. Up ahead, a fallen log blocks the path.",
      createdAt: "2025-03-01T14:30:00.000Z",
    },
    {
      id: LOG_IDS[5],
      sessionId: SESSION_ID,
      sceneId: SCENE_AMBUSH,
      type: "npc",
      text: "Happs: 'Well, well. Pay the toll or leave your gear. Your choice.'",
      createdAt: "2025-03-01T14:31:00.000Z",
    },
    {
      id: LOG_IDS[6],
      sessionId: SESSION_ID,
      sceneId: SCENE_AMBUSH,
      type: "player",
      text: "We're not paying. Roll initiative.",
      createdAt: "2025-03-01T14:32:00.000Z",
    },
    {
      id: LOG_IDS[7],
      sessionId: SESSION_ID,
      sceneId: SCENE_AMBUSH,
      type: "system",
      text: "Combat started. Happs Bydon and 3 bandits.",
      createdAt: "2025-03-01T14:32:00.000Z",
    },
    {
      id: LOG_IDS[8],
      sessionId: SESSION_ID,
      sceneId: SCENE_THORN,
      type: "narration",
      text: "The Thorn River glints in the afternoon light. The scout points to fresh tracks along the bank.",
      createdAt: "2025-03-01T15:00:00.000Z",
    },
    {
      id: LOG_IDS[9],
      sessionId: SESSION_ID,
      sceneId: SCENE_THORN,
      type: "npc",
      text: "Scout: 'Something big came through here. Not human. Maybe an hour ago.'",
      createdAt: "2025-03-01T15:01:00.000Z",
    },
  ],
  narrationClips: [
    {
      id: CLIP_IDS[0],
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      title: "Arrival read-aloud",
      voiceId: VOICE_WARM,
      duration: 12,
      createdAt: "2025-03-01T14:05:00.000Z",
    },
    {
      id: CLIP_IDS[1],
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      sceneId: SCENE_ARRIVAL,
      title: "Oleg's greeting",
      voiceId: VOICE_NOBLE,
      duration: 8,
      createdAt: "2025-03-01T14:07:00.000Z",
    },
    {
      id: CLIP_IDS[2],
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      sceneId: SCENE_AMBUSH,
      title: "Happs's demand",
      voiceId: VOICE_BANDIT,
      duration: 6,
      createdAt: "2025-03-01T14:31:00.000Z",
    },
    {
      id: CLIP_IDS[3],
      campaignId: CAMPAIGN_ID,
      sessionId: SESSION_ID,
      sceneId: SCENE_THORN,
      title: "Thorn River atmosphere",
      voiceId: VOICE_WARM,
      duration: 15,
      createdAt: "2025-03-01T15:00:00.000Z",
    },
  ],
};
