"""
Co-DM Live Board — Gradio app mounted at /live on the FastAPI server.

Visual design matches test_ui.html (parchment + dark theme).
Backend: TTS via tts_service.generate(), Co-GM via ai_service.generate_dialogue().
"""
from __future__ import annotations

import html
import logging

import gradio as gr

log = logging.getLogger(__name__)

# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=IM+Fell+English:ital@0;1&display=swap');

:root {
  --p-base:    #e8d8a8;
  --p-light:   #f4ead0;
  --p-mid:     #d8c490;
  --p-dark:    #c4aa70;
  --p-deep:    #a88a50;
  --ink:       #2e2010;
  --ink-lt:    #4a3420;
  --ink-faint: #7a6040;
  --gold:      #b8903c;
  --gold-lt:   #d4aa5a;
  --gold-glow: rgba(184,144,60,0.4);
  --red:       #7a2020;
  --red-lt:    #a03030;
  --blue:      #2a4060;
  --green:     #2a5020;
  --amber:     #7a5010;
  --shadow:    rgba(20,12,4,0.35);
}

/* ── Override Gradio defaults ──────────────────────────────── */
body, .gradio-container {
  background: #0e0a06 !important;
  background-image:
    radial-gradient(ellipse 80% 60% at 50% 0%, #1c1408 0%, transparent 70%),
    radial-gradient(ellipse 60% 80% at 0% 50%,  #120e06 0%, transparent 60%) !important;
  font-family: 'IM Fell English', Georgia, serif;
  color: var(--ink);
}
.gradio-container > .main, footer { display: none !important; }
#live-board-root { display: block !important; width: 100%; }

/* ── Parchment panels ──────────────────────────────────────── */
.lb-panel {
  background: var(--p-light);
  background-image: linear-gradient(170deg, var(--p-light) 0%, var(--p-base) 50%, var(--p-mid) 100%);
  border: 1px solid var(--p-dark);
  border-radius: 4px;
  overflow: hidden;
}

.panel-chrome {
  background: linear-gradient(180deg, #1c1408 0%, #2c2010 50%, #1c1408 100%);
  border-bottom: 1px solid var(--gold);
  padding: 5px 12px;
  font-family: 'Cinzel', serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.2em;
  color: var(--gold);
  text-align: center;
  text-transform: uppercase;
}
.rune { opacity: 0.55; margin: 0 4px; }

/* ── Scroll header ─────────────────────────────────────────── */
.scroll-header {
  background: linear-gradient(180deg, #1c1408 0%, #2c2010 50%, #1c1408 100%);
  border-bottom: 2px solid var(--gold);
  padding: 0 16px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.rune-strip {
  font-family: 'Cinzel', serif;
  font-size: 10px;
  letter-spacing: 5px;
  color: var(--gold);
  opacity: 0.55;
  white-space: nowrap;
}
.header-center { text-align: center; }
.header-subtitle {
  font-family: 'Cinzel', serif;
  font-size: 9px;
  letter-spacing: 0.35em;
  color: var(--gold-lt);
  opacity: 0.8;
  text-transform: uppercase;
}
.header-title {
  font-family: 'Cinzel', serif;
  font-size: 22px;
  font-weight: 900;
  color: var(--gold);
  letter-spacing: 0.12em;
  text-shadow: 0 0 16px var(--gold-glow), 0 2px 4px rgba(0,0,0,0.5);
}
.header-title em { font-style: normal; color: var(--gold-lt); }
.header-emblem { font-size: 13px; margin: 0 4px; opacity: 0.7; }

/* ── Campaign banner ──────────────────────────────────────── */
.campaign-title-bar {
  background: linear-gradient(90deg, #2a1c08 0%, #3a2a10 50%, #2a1c08 100%);
  padding: 6px 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  border-bottom: 1px solid var(--gold);
}
.campaign-label {
  font-family: 'Cinzel', serif;
  font-size: 9px;
  letter-spacing: 0.3em;
  color: var(--p-mid);
  opacity: 0.7;
}
.campaign-name {
  font-family: 'Cinzel', serif;
  font-size: 13px;
  font-weight: 700;
  color: var(--gold);
  letter-spacing: 0.15em;
  text-shadow: 0 0 8px var(--gold-glow);
}
.scene-art {
  position: relative;
  overflow: hidden;
  height: 140px;
  background: linear-gradient(180deg, #8fa8c0 0%, #9bb0c4 15%, #b0c0d0 30%,
    #c8ccc0 50%, #b0a890 65%, #8a7860 80%, #6a5840 100%);
}
.scene-art svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.scene-caption {
  background: linear-gradient(90deg, #1e1408 0%, #2e2010 50%, #1e1408 100%);
  padding: 5px 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  border-top: 1px solid var(--gold);
}
.scene-caption-text {
  font-family: 'Cinzel', serif;
  font-size: 9px;
  letter-spacing: 0.25em;
  color: var(--p-mid);
  opacity: 0.8;
}
.caption-divider { width: 1px; height: 10px; background: var(--gold); opacity: 0.3; }
.caption-highlight { color: var(--gold-lt); opacity: 0.9; }
.live-dot {
  display: inline-block;
  width: 6px; height: 6px;
  background: #4a9a4a;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
  will-change: opacity;
  animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Character roster ─────────────────────────────────────── */
.roster-scroll { padding: 8px; overflow-y: auto; max-height: 340px; }
.char-card {
  display: flex;
  gap: 8px;
  padding: 6px;
  border-radius: 4px;
  margin-bottom: 4px;
  background: rgba(255,255,255,0.25);
  border: 1px solid var(--p-dark);
  transition: background 0.2s;
}
.char-card.hurt    { background: rgba(122,32,32,0.08); border-color: var(--red-lt); }
.char-card.low     { background: rgba(122,32,32,0.14); border-color: var(--red); }
.portrait-frame {
  width: 44px; height: 52px;
  border-radius: 3px;
  overflow: hidden;
  border: 1px solid var(--gold);
  flex-shrink: 0;
  background: var(--p-mid);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}
.portrait-frame.gold-border { border-color: var(--gold-lt); border-width: 2px; }
.portrait-img { width: 100%; height: 100%; object-fit: cover; border-radius: inherit; display: block; }
.char-info { flex: 1; min-width: 0; }
.char-name { font-family: 'Cinzel', serif; font-size: 12px; font-weight: 700; color: var(--ink); }
.char-class { font-size: 9px; color: var(--ink-faint); margin-bottom: 3px; }
.char-status { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; margin-bottom: 4px; }
.status-icon { display: inline-flex; align-items: center; }
.heart-icon { width: 12px; height: 12px; display: inline-block; }
.heart-red   { fill: #c44444; }
.heart-amber { fill: #b88000; }
.heart-dark  { fill: #555; }
.status-badge {
  font-family: 'Cinzel', serif;
  font-size: 8px;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 2px;
  letter-spacing: 0.05em;
}
.sb-hp   { background: rgba(122,32,32,0.15); color: var(--red); border: 1px solid rgba(122,32,32,0.3); }
.sb-ac   { background: rgba(42,64,96,0.15);  color: var(--blue); border: 1px solid rgba(42,64,96,0.3); }
.sb-gold { background: rgba(184,144,60,0.15); color: var(--gold); border: 1px solid rgba(184,144,60,0.3); }
.sb-bad  { background: rgba(122,32,32,0.2);  color: var(--red-lt); border: 1px solid rgba(122,32,32,0.4); }
.hp-track { height: 5px; background: rgba(20,12,4,0.15); border-radius: 3px; overflow: hidden; margin-top: 2px; }
.hp-fill  { height: 100%; border-radius: 3px; transition: width 0.4s; }
.hp-hi    { background: linear-gradient(90deg, #4a9a4a, #6ab86a); }
.hp-mid   { background: linear-gradient(90deg, #b8800a, #d4a020); }
.hp-lo    { background: linear-gradient(90deg, #8a2020, #c43030); }
.ink-div  { height: 1px; background: linear-gradient(90deg, transparent, var(--p-dark), transparent); margin: 3px 0; }

/* ── Tool cards ───────────────────────────────────────────── */
.tools-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  padding: 8px;
}
.tool-card {
  position: relative;
  overflow: hidden;
  height: 70px;
  border-radius: 4px;
  border: 1px solid var(--p-dark);
  background: var(--p-mid);
  cursor: pointer;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  transition: transform 0.15s, box-shadow 0.15s;
}
.tool-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px var(--shadow); }
.tool-card.magic { border-color: var(--blue); }
.tool-card.danger { border-color: var(--red); }
.tool-icon-img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: 0; }
.t-label {
  position: relative; z-index: 1;
  font-family: 'Cinzel', serif;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.1em;
  color: #fff;
  background: rgba(20,12,4,0.65);
  padding: 3px 6px;
  border-radius: 2px;
  text-transform: uppercase;
  margin-bottom: 4px;
  text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}

/* ── Encounter tracker ────────────────────────────────────── */
.encounter-body { display: flex; gap: 0; height: 100%; }
.init-panel { width: 140px; flex-shrink: 0; border-right: 1px solid var(--p-dark); display: flex; flex-direction: column; }
.init-header {
  font-family: 'Cinzel', serif;
  font-size: 9px;
  letter-spacing: 0.2em;
  color: var(--gold);
  padding: 5px 8px;
  border-bottom: 1px solid var(--p-dark);
  background: rgba(20,12,4,0.08);
}
.init-scroll { flex: 1; overflow-y: auto; }
.init-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  cursor: pointer;
  border-bottom: 1px solid rgba(196,170,112,0.2);
  font-size: 11px;
  transition: background 0.15s;
}
.init-row:hover { background: rgba(184,144,60,0.08); }
.init-row.on { background: rgba(184,144,60,0.2); font-weight: 700; }
.init-row.enemy { background: rgba(122,32,32,0.08); }
.init-row.enemy.on { background: rgba(122,32,32,0.2); }
.init-num { font-family: 'Cinzel', serif; font-size: 9px; color: var(--ink-faint); width: 12px; }
.init-roll { font-family: 'Cinzel', serif; font-size: 10px; font-weight: 700; color: var(--gold); width: 20px; }
.init-name { flex: 1; font-size: 10px; color: var(--ink); }
.init-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-green  { background: var(--green); }
.dot-red    { background: var(--red); }
.dot-amber  { background: var(--amber); }
.dot-red-lt { background: var(--red-lt); }
.init-controls { display: flex; gap: 4px; padding: 5px; border-top: 1px solid var(--p-dark); }
.ic-btn {
  flex: 1;
  font-family: 'Cinzel', serif;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 4px 2px;
  border: 1px solid var(--p-dark);
  border-radius: 3px;
  background: rgba(255,255,255,0.4);
  color: var(--ink);
  cursor: pointer;
  transition: background 0.15s;
}
.ic-btn:hover { background: rgba(184,144,60,0.2); }
.ic-btn.next  { border-color: var(--gold); color: var(--gold); }
.init-empty {
  padding: 18px 10px;
  text-align: center;
  font-family: 'Cinzel', serif;
  font-size: 10px;
  color: var(--ink-faint);
  letter-spacing: 0.1em;
}

/* ── Battle grid ──────────────────────────────────────────── */
.grid-panel { flex: 1; display: flex; flex-direction: column; padding: 6px; }
.battle-grid {
  flex: 1;
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  gap: 2px;
  border: 1px solid var(--p-dark);
  border-radius: 3px;
  overflow: hidden;
  background: rgba(20,12,4,0.05);
}
.gcell {
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-family: 'Cinzel', serif;
  font-weight: 700;
  cursor: pointer;
  border: 1px solid rgba(196,170,112,0.15);
  background: rgba(255,255,255,0.1);
  color: var(--ink);
  transition: background 0.1s;
  user-select: none;
}
.gcell:hover { background: rgba(184,144,60,0.15); }
.gcell.pc   { background: rgba(42,80,32,0.25); color: var(--green); }
.gcell.npc  { background: rgba(122,32,32,0.25); color: var(--red); }
.gcell.wall { background: rgba(20,12,4,0.4); color: var(--ink-faint); cursor: default; }
.gcell.sel  { background: rgba(184,144,60,0.35); outline: 2px solid var(--gold); }
.grid-toolbar { display: flex; gap: 4px; margin-top: 5px; }
.gt-btn {
  flex: 1;
  font-family: 'Cinzel', serif;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 4px;
  border: 1px solid var(--p-dark);
  border-radius: 3px;
  background: rgba(255,255,255,0.4);
  color: var(--ink);
  cursor: pointer;
  transition: background 0.15s;
}
.gt-btn.active { background: rgba(184,144,60,0.3); border-color: var(--gold); color: var(--gold); }

/* ── World map ────────────────────────────────────────────── */
.map-art { position: relative; overflow: hidden; height: 110px; }
.map-art svg { width: 100%; height: 100%; }
.map-city-label {
  position: absolute;
  font-family: 'Cinzel', serif;
  font-size: 7px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: 0.05em;
  text-shadow: 0 1px 2px rgba(255,255,255,0.6);
  pointer-events: none;
  white-space: nowrap;
}
.map-lbl-1 { left: 8%;  top: 68%; }
.map-lbl-2 { left: 44%; top: 38%; }
.map-lbl-3 { left: 70%; top: 48%; }
.map-lbl-4 { left: 30%; top: 78%; font-size: 6px; opacity: 0.7; }
.map-pin   { position: absolute; pointer-events: none; }
.map-pin-1 { left: 43%; top: 26%; }
.map-pin-svg { width: 12px; height: 16px; display: block; }
.map-title {
  position: absolute; bottom: 4px; right: 6px;
  font-family: 'Cinzel', serif;
  font-size: 8px;
  color: var(--ink-faint);
  font-style: italic;
  pointer-events: none;
}
.map-footer {
  display: flex;
  gap: 3px;
  padding: 4px 6px;
  background: rgba(20,12,4,0.06);
  border-top: 1px solid var(--p-dark);
}
.map-btn {
  flex: 1;
  font-family: 'Cinzel', serif;
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-align: center;
  padding: 3px 4px;
  border: 1px solid var(--p-dark);
  border-radius: 2px;
  background: rgba(255,255,255,0.4);
  color: var(--ink);
  cursor: pointer;
}
.map-btn:hover { background: rgba(184,144,60,0.2); }

/* ── Spell book ───────────────────────────────────────────── */
.spellbook-body { display: flex; gap: 0; overflow: hidden; }
.spell-slots-col { width: 90px; flex-shrink: 0; border-right: 1px solid var(--p-dark); padding: 6px; }
.slot-lbl {
  font-family: 'Cinzel', serif;
  font-size: 7.5px;
  font-weight: 700;
  letter-spacing: 0.15em;
  color: var(--gold);
  margin-bottom: 4px;
  text-transform: uppercase;
}
.slot-row { display: flex; align-items: center; gap: 3px; margin-bottom: 3px; }
.slot-l { font-family: 'Cinzel', serif; font-size: 8px; color: var(--ink-faint); width: 24px; }
.pip {
  width: 9px; height: 9px;
  border-radius: 50%;
  border: 1.5px solid var(--gold);
  background: rgba(184,144,60,0.3);
  cursor: pointer;
  transition: background 0.15s;
}
.pip.spent { background: var(--p-mid); border-color: var(--ink-faint); }
.pip:hover { background: rgba(184,144,60,0.6); }
.dc-val { font-family: 'Cinzel', serif; font-size: 20px; font-weight: 700; color: var(--red); text-align: center; }
.dc-sub { font-family: 'Cinzel', serif; font-size: 8px; color: var(--ink-faint); text-align: center; letter-spacing: 0.1em; }
.spell-log { flex: 1; overflow-y: auto; padding: 6px; }
.spell-entry { margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid rgba(196,170,112,0.3); }
.spell-entry-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px; }
.spell-entry-name { font-family: 'Cinzel', serif; font-size: 9px; font-weight: 700; color: var(--ink); }
.spell-entry-time { font-size: 8px; color: var(--ink-faint); font-style: italic; }
.spell-entry-desc { font-size: 9px; color: var(--ink-lt); line-height: 1.4; font-style: italic; }

/* ── Notes ────────────────────────────────────────────────── */
textarea.notes-ta {
  width: 100%;
  background: rgba(255,255,255,0.3);
  border: 1px solid var(--p-dark);
  border-radius: 3px;
  padding: 8px;
  font-family: 'IM Fell English', Georgia, serif;
  font-size: 11px;
  color: var(--ink);
  resize: none;
  outline: none;
  line-height: 1.5;
}
textarea.notes-ta:focus { border-color: var(--gold); box-shadow: 0 0 6px var(--gold-glow); }

/* ── Gradio component overrides ───────────────────────────── */
.lb-section label { font-family: 'Cinzel', serif !important; font-size: 10px !important;
                    letter-spacing: 0.1em !important; color: var(--gold) !important;
                    text-transform: uppercase !important; }
.lb-section input, .lb-section textarea, .lb-section select {
  background: rgba(255,255,255,0.4) !important;
  border: 1px solid var(--p-dark) !important;
  border-radius: 3px !important;
  color: var(--ink) !important;
  font-family: 'IM Fell English', Georgia, serif !important;
}
.lb-section input:focus, .lb-section textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 6px var(--gold-glow) !important;
}
.lb-btn {
  font-family: 'Cinzel', serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  background: rgba(184,144,60,0.2) !important;
  border: 1px solid var(--gold) !important;
  color: var(--ink) !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  text-transform: uppercase !important;
}
.lb-btn:hover { background: rgba(184,144,60,0.4) !important; }
.lb-btn-primary { background: rgba(184,144,60,0.35) !important; }

/* ── Co-GM dialogue log ───────────────────────────────────── */
.dialogue-entry {
  padding: 6px 10px;
  border-radius: 4px;
  margin-bottom: 6px;
  font-size: 11px;
  line-height: 1.5;
}
.dialogue-npc {
  background: rgba(42,64,96,0.12);
  border-left: 3px solid var(--blue);
  color: var(--ink);
}
.dialogue-gm {
  background: rgba(184,144,60,0.08);
  border-left: 3px solid var(--gold);
  color: var(--ink-lt);
}
.dialogue-speaker { font-family: 'Cinzel', serif; font-size: 9px; font-weight: 700;
                    color: var(--gold); letter-spacing: 0.1em; display: block; margin-bottom: 2px; }
"""

# ─── HTML constants ───────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="scroll-header">
  <span class="rune-strip">ᚠ ᚢ ᚦ ᚨ ᚱ ᚲ ᚷ ᚹ ᚺ ᚾ</span>
  <div class="header-center">
    <div class="header-subtitle">High Fantasy D&amp;D</div>
    <div class="header-title">
      <span class="header-emblem">⚜</span>
      Co-<em>DM</em> Edition
      <span class="header-emblem">⚜</span>
    </div>
  </div>
  <span class="rune-strip">ᛁ ᛃ ᛇ ᛈ ᛉ ᛊ ᛏ ᛒ ᛖ ᛗ</span>
</div>
"""

CAMPAIGN_BANNER_HTML = """
<div style="border-bottom:1px solid var(--p-dark); border-right:1px solid var(--p-dark);">
  <div class="campaign-title-bar">
    <span class="campaign-label">CURRENT CAMPAIGN:</span>
    <span class="campaign-name">THE SHATTERED CROWN</span>
  </div>
  <div class="scene-art">
    <svg viewBox="0 0 760 170" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stop-color="#6080a0"/>
          <stop offset="40%"  stop-color="#8098b8"/>
          <stop offset="70%"  stop-color="#a0b0c0"/>
          <stop offset="100%" stop-color="#b8b0a0"/>
        </linearGradient>
        <linearGradient id="fog" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stop-color="#c8d0d8" stop-opacity="0.0"/>
          <stop offset="100%" stop-color="#d8d0c0" stop-opacity="0.55"/>
        </linearGradient>
        <filter id="blr"><feGaussianBlur stdDeviation="2"/></filter>
        <filter id="blr2"><feGaussianBlur stdDeviation="1"/></filter>
      </defs>
      <rect width="760" height="170" fill="url(#sky)"/>
      <polygon points="0,140 60,60 120,140"   fill="#7a8898" opacity="0.4"  filter="url(#blr)"/>
      <polygon points="50,140 130,45 210,140"  fill="#6a7888" opacity="0.35" filter="url(#blr)"/>
      <polygon points="140,140 210,30 280,140" fill="#788898" opacity="0.4"  filter="url(#blr)"/>
      <polygon points="220,140 290,55 360,140" fill="#6e7e8e" opacity="0.3"  filter="url(#blr)"/>
      <polygon points="300,140 375,25 450,140" fill="#788898" opacity="0.38" filter="url(#blr)"/>
      <polygon points="380,140 460,50 540,140" fill="#6a7888" opacity="0.3"  filter="url(#blr)"/>
      <polygon points="460,140 540,35 620,140" fill="#788898" opacity="0.35" filter="url(#blr)"/>
      <polygon points="550,140 630,60 700,140" fill="#6e7e90" opacity="0.3"  filter="url(#blr)"/>
      <polygon points="640,140 700,45 760,140" fill="#788898" opacity="0.35" filter="url(#blr)"/>
      <g opacity="0.45" filter="url(#blr2)" transform="translate(520,40)">
        <rect x="0"  y="60" width="14" height="60" fill="#4a4a50"/>
        <rect x="16" y="40" width="20" height="80" fill="#5a5a60"/>
        <rect x="38" y="55" width="14" height="65" fill="#4a4a50"/>
        <rect x="0"  y="56" width="4"  height="6"  fill="#4a4a50"/>
        <rect x="6"  y="56" width="4"  height="6"  fill="#4a4a50"/>
        <rect x="10" y="56" width="4"  height="6"  fill="#4a4a50"/>
        <rect x="16" y="36" width="5"  height="7"  fill="#5a5a60"/>
        <rect x="23" y="36" width="5"  height="7"  fill="#5a5a60"/>
        <rect x="30" y="36" width="5"  height="7"  fill="#5a5a60"/>
        <rect x="20" y="52" width="10" height="14" fill="#202040" rx="2"/>
        <polygon points="52,100 60,75 70,65 80,70 85,100" fill="#5a5860" opacity="0.7"/>
      </g>
      <g opacity="0.35" filter="url(#blr2)" transform="translate(30,60)">
        <rect x="4" y="40" width="3" height="60" fill="#3a2a1a"/>
        <line x1="5.5" y1="55" x2="-6"  y2="42" stroke="#3a2a1a" stroke-width="2"/>
        <line x1="5.5" y1="62" x2="18"  y2="50" stroke="#3a2a1a" stroke-width="1.5"/>
        <line x1="5.5" y1="70" x2="-4"  y2="65" stroke="#3a2a1a" stroke-width="1.5"/>
        <line x1="5.5" y1="75" x2="14"  y2="72" stroke="#3a2a1a" stroke-width="1.5"/>
      </g>
      <ellipse cx="380" cy="175" rx="440" ry="60" fill="#8a7858" opacity="0.6"/>
      <rect x="0" y="150" width="760" height="30" fill="#7a6848" opacity="0.7"/>
      <path d="M 200,175 Q 350,140 500,145 Q 600,148 660,160 L 760,170 L 0,170 Z" fill="#a09070" opacity="0.4"/>
      <rect width="760" height="170" fill="url(#fog)"/>
      <g transform="translate(180,90)" opacity="0.88">
        <rect x="0" y="12" width="12" height="28" rx="2" fill="#1e1408"/>
        <polygon points="-3,24 15,24 18,55 -6,55" fill="#1e1408"/>
        <ellipse cx="6" cy="10" rx="6" ry="7" fill="#1e1408"/>
        <polygon points="0,4 12,4 7,-14" fill="#1e1408"/>
        <rect x="0" y="2" width="12" height="3" fill="#1e1408"/>
        <rect x="-8" y="-8" width="2" height="68" fill="#2a1a08" rx="1"/>
        <ellipse cx="-7" cy="-10" rx="4" ry="4" fill="#3a2810"/>
      </g>
      <g transform="translate(240,95)" opacity="0.88">
        <rect x="0" y="10" width="11" height="25" rx="1" fill="#1a1208"/>
        <rect x="0"  y="33" width="5" height="22" rx="1" fill="#1a1208"/>
        <rect x="6"  y="33" width="5" height="22" rx="1" fill="#1a1208"/>
        <ellipse cx="5.5" cy="7" rx="5.5" ry="6" fill="#1a1208"/>
        <path d="M 0,4 Q 5.5,-4 11,4" fill="#1a1208"/>
        <path d="M 15,2 Q 22,10 15,28" stroke="#2a1a08" stroke-width="2" fill="none"/>
        <line x1="15" y1="2" x2="15" y2="28" stroke="#1a0e04" stroke-width="1"/>
        <rect x="-6" y="6" width="4" height="14" rx="1" fill="#2a1a08"/>
      </g>
      <g transform="translate(305,82)" opacity="0.9">
        <rect x="-1" y="14" width="16" height="28" rx="2" fill="#181008"/>
        <rect x="-5" y="12" width="6"  height="8"  rx="2" fill="#181008"/>
        <rect x="13" y="12" width="6"  height="8"  rx="2" fill="#181008"/>
        <rect x="0"  y="40" width="6"  height="26" rx="1" fill="#181008"/>
        <rect x="8"  y="40" width="6"  height="26" rx="1" fill="#181008"/>
        <rect x="0" y="0" width="14" height="14" rx="3" fill="#181008"/>
        <rect x="2" y="6" width="10" height="2" fill="#302010" opacity="0.7"/>
        <path d="M 7,0 Q 4,-12 8,-18 Q 12,-12 7,0" fill="#1e1408"/>
        <path d="M -14,14 L -6,14 L -6,38 Q -10,42 -14,38 Z" fill="#1e1008"/>
        <rect x="18" y="-6" width="3" height="36" rx="1" fill="#1a1208" transform="rotate(-18,20,15)"/>
        <rect x="13" y="14" width="12" height="3" rx="1" fill="#1a1208"/>
      </g>
      <g transform="translate(365,100)" opacity="0.85">
        <rect x="0" y="10" width="10" height="22" rx="1" fill="#141008"/>
        <rect x="0" y="30" width="5" height="18" rx="1" fill="#141008"/>
        <rect x="5" y="30" width="5" height="16" rx="1" fill="#141008" transform="rotate(8,8,30)"/>
        <ellipse cx="5" cy="7" rx="5" ry="6" fill="#141008"/>
        <path d="M 0,5 Q 5,-2 10,5" fill="#141008"/>
        <rect x="1" y="7" width="8" height="3" fill="#141008"/>
        <rect x="12" y="18" width="2" height="14" rx="1" fill="#1a1208" transform="rotate(-20,13,25)"/>
        <rect x="-4" y="18" width="2" height="12" rx="1" fill="#1a1208" transform="rotate(20,-3,24)"/>
      </g>
      <g transform="translate(420,93)" opacity="0.85">
        <rect x="0" y="12" width="12" height="26" rx="2" fill="#1a1210"/>
        <polygon points="-2,24 14,24 16,55 -4,55" fill="#1a1210"/>
        <ellipse cx="6" cy="9" rx="6" ry="7" fill="#1a1210"/>
        <path d="M -1,5 Q 6,-3 13,5 L 13,12 L -1,12 Z" fill="#1a1210"/>
        <rect x="14" y="-2" width="2" height="62" rx="1" fill="#1e1410"/>
        <rect x="10" y="-5" width="10" height="2" rx="1" fill="#1e1410"/>
        <rect x="14" y="-8" width="2" height="10" rx="1" fill="#1e1410"/>
        <rect x="-8" y="18" width="8" height="10" rx="1" fill="#1e1410"/>
      </g>
      <g fill="none" stroke="#2a3040" stroke-width="1.5" opacity="0.4">
        <path d="M100,30 Q103,26 107,30"/>
        <path d="M115,22 Q118,18 122,22"/>
        <path d="M600,35 Q603,31 607,35"/>
        <path d="M620,28 Q623,24 627,28"/>
        <path d="M640,40 Q643,36 647,40"/>
      </g>
    </svg>
  </div>
  <div class="scene-caption">
    <span class="scene-caption-text">NEXT SESSION: 2 DAYS AWAY</span>
    <div class="caption-divider"></div>
    <span class="scene-caption-text caption-highlight">THE SUNKEN TEMPLE</span>
    <div class="caption-divider"></div>
    <span class="scene-caption-text"><span class="live-dot"></span>SESSION ACTIVE</span>
  </div>
</div>
"""

ENCOUNTER_TRACKER_HTML = """
<div class="lb-panel" style="height:260px; display:flex; flex-direction:column;">
  <div class="panel-chrome"><span class="rune">⚔</span> ENCOUNTER TRACKER <span class="rune">⚔</span></div>
  <div class="encounter-body" style="flex:1; min-height:0;">
    <div class="init-panel">
      <div class="init-header">⚔ INITIATIVE</div>
      <div class="init-scroll" id="lb_initList">
        <div class="init-row on" onclick="lbSelectInit(this)">
          <span class="init-num">1</span><span class="init-roll">22</span>
          <span class="init-name">Aethelred</span><span class="init-dot dot-green"></span>
        </div>
        <div class="init-row enemy" onclick="lbSelectInit(this)">
          <span class="init-num">2</span><span class="init-roll">19</span>
          <span class="init-name">Beholder</span><span class="init-dot dot-red"></span>
        </div>
        <div class="init-row" onclick="lbSelectInit(this)">
          <span class="init-num">3</span><span class="init-roll">17</span>
          <span class="init-name">Torin</span><span class="init-dot dot-green"></span>
        </div>
        <div class="init-row" onclick="lbSelectInit(this)">
          <span class="init-num">4</span><span class="init-roll">15</span>
          <span class="init-name">Mira</span><span class="init-dot dot-green"></span>
        </div>
        <div class="init-row enemy" onclick="lbSelectInit(this)">
          <span class="init-num">5</span><span class="init-roll">13</span>
          <span class="init-name">Eye Tyrant</span><span class="init-dot dot-red"></span>
        </div>
        <div class="init-row" onclick="lbSelectInit(this)">
          <span class="init-num">6</span><span class="init-roll">11</span>
          <span class="init-name">Lira</span><span class="init-dot dot-amber"></span>
        </div>
        <div class="init-row" onclick="lbSelectInit(this)">
          <span class="init-num">7</span><span class="init-roll">6</span>
          <span class="init-name">Zephyr</span><span class="init-dot dot-red-lt"></span>
        </div>
      </div>
      <div class="init-controls">
        <button type="button" class="ic-btn next" onclick="lbNextInit()">▶ NEXT</button>
        <button type="button" class="ic-btn" onclick="lbAddInit()">+ ADD</button>
        <button type="button" class="ic-btn" onclick="lbEndCombat()">⏹ END</button>
      </div>
    </div>
    <div class="grid-panel">
      <div class="battle-grid" id="lb_battleGrid"></div>
      <div class="grid-toolbar">
        <button type="button" class="gt-btn active" onclick="lbSetTool(this,'move')">Move</button>
        <button type="button" class="gt-btn" onclick="lbSetTool(this,'place')">Place</button>
        <button type="button" class="gt-btn" onclick="lbSetTool(this,'wall')">Wall</button>
        <button type="button" class="gt-btn" onclick="lbClearGrid()">Clear</button>
      </div>
    </div>
  </div>
</div>
<script>
(function(){
  const ROWS=6,COLS=9;
  const TOKENS={2:'Pal',11:'Rog',19:'Wiz',28:'Clr',38:'Drd',5:'Eye',14:'Beh',
                8:'▪',17:'▪',26:'▪',24:'≈',25:'≈'};
  const CLS={'Eye':'npc','Beh':'npc','▪':'wall','≈':'wall'};
  let gridTool='move',sel=null,curInit=null;

  function buildGrid(){
    const g=document.getElementById('lb_battleGrid');
    if(!g)return;
    g.innerHTML='';
    for(let i=0;i<ROWS*COLS;i++){
      const c=document.createElement('div');
      c.className='gcell';
      const t=TOKENS[i];
      if(t){c.textContent=t;c.classList.add(CLS[t]||'pc');}
      c.dataset.i=i;
      c.addEventListener('click',onCell);
      g.appendChild(c);
    }
  }
  function onCell(e){
    const cells=document.querySelectorAll('#lb_battleGrid .gcell');
    const idx=parseInt(e.currentTarget.dataset.i);
    if(gridTool==='wall'){
      const c=cells[idx];
      if(!c.classList.contains('pc')&&!c.classList.contains('npc')){
        if(c.classList.contains('wall')){c.textContent='';c.classList.remove('wall');}
        else{c.textContent='▪';c.classList.add('wall');}
      }return;
    }
    if(gridTool==='move'){
      if(sel!==null){
        const from=cells[sel],to=cells[idx];
        if(!to.classList.contains('wall')&&!to.classList.contains('pc')&&!to.classList.contains('npc')){
          to.textContent=from.textContent;to.className=from.className;
          to.classList.remove('sel');from.textContent='';from.className='gcell';
        }
        cells[sel].classList.remove('sel');sel=null;
      }else{
        const c=cells[idx];
        if(c.textContent&&!c.classList.contains('wall')){sel=idx;c.classList.add('sel');}
      }
    }
  }
  window.lbSetTool=function(btn,t){
    gridTool=t;sel=null;
    document.querySelectorAll('#lb_battleGrid .gcell').forEach(c=>c.classList.remove('sel'));
    document.querySelectorAll('.gt-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
  };
  window.lbClearGrid=function(){
    if(confirm('Clear all tokens?')){Object.keys(TOKENS).forEach(k=>delete TOKENS[k]);buildGrid();}
  };
  window.lbSelectInit=function(el){
    document.querySelectorAll('#lb_initList .init-row').forEach(r=>r.classList.remove('on'));
    el.classList.add('on');curInit=el;
  };
  window.lbNextInit=function(){
    const rows=[...document.querySelectorAll('#lb_initList .init-row')];
    const idx=rows.indexOf(curInit??rows[0]);
    const next=rows[(idx+1)%rows.length];
    window.lbSelectInit(next);next.scrollIntoView({block:'nearest',behavior:'smooth'});
  };
  window.lbAddInit=function(){
    const name=prompt('Combatant name:');if(!name)return;
    const roll=Math.floor(Math.random()*20)+1;
    const isEnemy=/orc|troll|goblin|beholder|undead|wraith|golem/i.test(name);
    const list=document.getElementById('lb_initList');
    const row=document.createElement('div');
    row.className='init-row'+(isEnemy?' enemy':'');
    const num=list.querySelectorAll('.init-row').length+1;
    row.innerHTML=`<span class="init-num">${num}</span><span class="init-roll">${roll}</span><span class="init-name">${name}</span><span class="init-dot ${isEnemy?'dot-red':'dot-green'}"></span>`;
    row.addEventListener('click',()=>window.lbSelectInit(row));
    const existing=[...list.querySelectorAll('.init-row')];
    let inserted=false;
    for(const el of existing){
      if(roll>parseInt(el.querySelector('.init-roll').textContent)){list.insertBefore(row,el);inserted=true;break;}
    }
    if(!inserted)list.appendChild(row);
  };
  window.lbEndCombat=function(){
    if(confirm('End combat and clear initiative?')){
      document.getElementById('lb_initList').innerHTML='<div class="init-empty">— No active encounter —</div>';
      curInit=null;
    }
  };
  buildGrid();
})();
</script>
"""

SPELLBOOK_HTML = """
<div class="lb-panel">
  <div class="panel-chrome"><span class="rune">✦</span> SPELL BOOK VIEWER — TORIN <span class="rune">✦</span></div>
  <div class="spellbook-body" style="height:160px;">
    <div class="spell-slots-col">
      <div class="slot-lbl">Spell Slots</div>
      <div class="slot-row"><span class="slot-l">Lv 1</span>
        <span class="pip"></span><span class="pip"></span>
        <span class="pip spent"></span><span class="pip spent"></span></div>
      <div class="slot-row"><span class="slot-l">Lv 2</span>
        <span class="pip"></span><span class="pip spent"></span><span class="pip spent"></span></div>
      <div class="slot-row"><span class="slot-l">Lv 3</span>
        <span class="pip"></span><span class="pip"></span><span class="pip spent"></span></div>
      <div class="slot-row"><span class="slot-l">Lv 4</span>
        <span class="pip"></span><span class="pip spent"></span></div>
      <div class="slot-row"><span class="slot-l">Lv 5</span>
        <span class="pip"></span></div>
      <div class="ink-div"></div>
      <div class="slot-lbl">Spell DC</div>
      <div class="dc-val">16</div>
      <div class="dc-sub">ATK +8</div>
    </div>
    <div class="spell-log">
      <div class="spell-entry">
        <div class="spell-entry-head">
          <span class="spell-entry-name">Fireball — 3rd Level</span>
          <span class="spell-entry-time">12 min ago</span>
        </div>
        <div class="spell-entry-desc">Cast at the Beholder cluster. DC 16 DEX save. 28 dmg (8d6). Two cultists failed.</div>
      </div>
      <div class="spell-entry">
        <div class="spell-entry-head">
          <span class="spell-entry-name">Shield — Reaction</span>
          <span class="spell-entry-time">4 min ago</span>
        </div>
        <div class="spell-entry-desc">Used reaction to deflect ray of disintegration. AC raised to 18. Slot expended.</div>
      </div>
    </div>
  </div>
</div>
<script>
document.querySelectorAll('.pip').forEach(p=>p.addEventListener('click',()=>p.classList.toggle('spent')));
</script>
"""

QUICK_TOOLS_HTML = """
<div class="lb-panel">
  <div class="panel-chrome"><span class="rune">⚔</span> QUICK TOOLS <span class="rune">⚔</span></div>
  <div class="tools-grid">
    <div class="tool-card" onclick="lbRollDice(20)">
      <img src="/static/img/tools/dice.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Roll Dice</span>
    </div>
    <div class="tool-card" onclick="alert('Monster Bestiary')">
      <img src="/static/img/tools/bestiary.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Monster Bestiary</span>
    </div>
    <div class="tool-card magic" onclick="alert('Spell Reference')">
      <img src="/static/img/tools/bluff.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Spell Ref</span>
    </div>
    <div class="tool-card" onclick="alert('Random NPC generated!')">
      <img src="/static/img/tools/npc.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Gen NPC</span>
    </div>
    <div class="tool-card" onclick="alert('Loot table')">
      <img src="/static/img/tools/loot.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Loot Table</span>
    </div>
    <div class="tool-card danger" onclick="lbApplyDamage()">
      <img src="/static/img/tools/damage.jpg" class="tool-icon-img" alt="" onerror="this.style.display='none'" />
      <span class="t-label">Apply Damage</span>
    </div>
  </div>
</div>
<script>
window.lbRollDice=function(n){
  const r=Math.floor(Math.random()*n)+1;
  const msg=r===n?`NATURAL ${n}!`:r===1?'Critical Fail!':`Result: ${r}`;
  alert(`d${n} → ${msg}`);
};
window.lbApplyDamage=function(){
  const d=prompt('Damage amount:');
  if(d&&!isNaN(d))alert(`${d} damage applied.`);
};
</script>
"""

WORLD_MAP_HTML = """
<div class="map-section">
  <div class="panel-chrome" style="padding:4px 10px;font-size:9px;">
    <span class="rune">ᛗ</span> WORLD MAP <span class="rune">ᛗ</span>
  </div>
  <div class="map-art">
    <svg viewBox="0 0 250 120" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
      <rect width="250" height="120" fill="#c8b070" opacity="0.5"/>
      <path d="M170,70 Q200,80 250,90 L250,120 L170,120 Z" fill="#7090a0" opacity="0.35"/>
      <polygon points="0,100 20,55 40,100"  fill="#8a7858" opacity="0.6"/>
      <polygon points="25,100 45,45 65,100" fill="#9a8868" opacity="0.55"/>
      <polygon points="50,100 65,60 80,100" fill="#8a7858" opacity="0.5"/>
      <ellipse cx="120" cy="70" rx="18" ry="12" fill="#4a6a3a" opacity="0.55"/>
      <ellipse cx="138" cy="66" rx="16" ry="13" fill="#3a5a2a" opacity="0.5"/>
      <ellipse cx="155" cy="72" rx="14" ry="11" fill="#4a6a3a" opacity="0.45"/>
      <path d="M0,75 Q50,65 90,78 Q130,90 170,72 Q210,54 250,60" stroke="#5080a0" stroke-width="2" fill="none" opacity="0.5"/>
      <path d="M30,110 Q80,90 125,82 Q170,74 210,88" stroke="#a09060" stroke-width="1.5" fill="none" opacity="0.5" stroke-dasharray="4,3"/>
      <circle cx="45"  cy="108" r="3.5" fill="#7a2020" opacity="0.8"/>
      <circle cx="125" cy="82"  r="3"   fill="#2a4060" opacity="0.8"/>
      <circle cx="195" cy="65"  r="3.5" fill="#7a2020" opacity="0.8"/>
      <circle cx="100" cy="55"  r="2.5" fill="#5a4020" opacity="0.7"/>
      <g transform="translate(111,32)" fill="#3a2810" opacity="0.8">
        <rect x="4" y="5" width="6" height="8"/>
        <rect x="3" y="3" width="2" height="3"/>
        <rect x="6.5" y="3" width="2" height="3"/>
        <rect x="10" y="3" width="2" height="3"/>
        <rect x="0" y="7" width="4" height="6"/>
        <rect x="-1" y="5" width="2" height="3"/>
        <rect x="2"  y="5" width="2" height="3"/>
        <rect x="10" y="7" width="4" height="6"/>
        <rect x="10" y="5" width="2" height="3"/>
        <rect x="13" y="5" width="2" height="3"/>
      </g>
    </svg>
    <div class="map-city-label map-lbl-1">Greyhaven</div>
    <div class="map-city-label map-lbl-2">Shattered Crown</div>
    <div class="map-city-label map-lbl-3">Sea Port</div>
    <div class="map-city-label map-lbl-4">Fantasy City</div>
    <div class="map-pin map-pin-1">
      <svg viewBox="0 0 12 16" class="map-pin-svg" aria-hidden="true">
        <path d="M6 0C3.24 0 1 2.24 1 5c0 3.75 5 11 5 11s5-7.25 5-11C11 2.24 8.76 0 6 0zm0 7.5C4.62 7.5 3.5 6.38 3.5 5S4.62 2.5 6 2.5 8.5 3.62 8.5 5 7.38 7.5 6 7.5z" fill="#7a2020"/>
      </svg>
    </div>
    <div class="map-title">The Shattered Crown</div>
  </div>
  <div class="map-footer">
    <div class="map-btn">World</div>
    <div class="map-btn">Region</div>
    <div class="map-btn">Dungeon</div>
    <div class="map-btn">Travel</div>
  </div>
</div>
"""

# ─── Backend helpers ──────────────────────────────────────────────────────────

_CHARS = [
    {"name": "Aethelred", "cls": "Human · Paladin · Lv 8",   "max": 80, "img": "aethelred.jpg", "ac": 18, "gold_border": True},
    {"name": "Lira",      "cls": "Half-Elf · Rogue · Lv 8",  "max": 58, "img": "lira.jpg",      "ac": 15, "gold_border": False},
    {"name": "Torin",     "cls": "Gnome · Wizard · Lv 8",    "max": 44, "img": "torin.jpg",     "ac": 13, "gold_border": False},
    {"name": "Zephyr",    "cls": "Wood Elf · Druid · Lv 8",  "max": 56, "img": "zephyr.jpg",    "ac": 14, "gold_border": False},
    {"name": "Mira",      "cls": "Human · Cleric · Lv 8",    "max": 55, "img": "mira.jpg",      "ac": 17, "gold_border": False},
]
_CHAR_DEFAULT_HP = {c["name"]: c["max"] for c in _CHARS}


def _hp_pct(hp: int, max_hp: int) -> float:
    return max(0.0, min(1.0, hp / max_hp)) if max_hp else 0.0


def _render_party_roster(hp_vals: dict) -> str:
    HEART = '<svg viewBox="0 0 24 24" class="heart-icon {cls}" aria-hidden="true"><path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z"/></svg>'
    rows = []
    for c in _CHARS:
        name = c["name"]
        hp = int(hp_vals.get(name, c["max"]))
        pct = _hp_pct(hp, c["max"])
        if pct > 0.6:
            card_cls, fill_cls, heart_cls = "", "hp-hi", "heart-red"
        elif pct > 0.25:
            card_cls, fill_cls, heart_cls = "hurt", "hp-mid", "heart-amber"
        else:
            card_cls, fill_cls, heart_cls = "low", "hp-lo", "heart-dark"
        border_cls = " gold-border" if c["gold_border"] else ""
        img_src = f"/static/img/portraits/{c['img']}"
        fb_url = f"https://api.dicebear.com/9.x/adventurer/svg?seed={name}"
        rows.append(f"""
<div class="char-card {card_cls}">
  <div class="portrait-frame{border_cls}">
    <img src="{img_src}" onerror="this.src='{fb_url}'" class="portrait-img" alt="{html.escape(name)}" />
  </div>
  <div class="char-info">
    <div class="char-name">{html.escape(name)}</div>
    <div class="char-class">{html.escape(c['cls'])}</div>
    <div class="char-status">
      <span class="status-icon">{HEART.format(cls=heart_cls)}</span>
      <span class="status-badge sb-hp">{hp}/{c['max']}</span>
      <span class="status-badge sb-ac">AC {c['ac']}</span>
    </div>
    <div class="hp-track"><div class="hp-fill {fill_cls}" style="width:{pct*100:.0f}%"></div></div>
  </div>
</div>""")
    return (
        '<div class="panel-chrome"><span class="rune">ᚨ</span> PARTY ROSTER <span class="rune">ᚨ</span></div>'
        '<div class="roster-scroll">' + "".join(rows) + "</div>"
    )


def _render_dialogue_log(history: list) -> str:
    if not history:
        return '<p style="color:var(--ink-faint);font-size:11px;text-align:center;padding:12px;">No dialogue yet.</p>'
    parts = []
    for msg in history[-10:]:  # show last 10 turns
        role = msg.get("role", "assistant")
        content = html.escape(msg.get("content", ""))
        if role == "assistant":
            parts.append(
                f'<div class="dialogue-entry dialogue-npc">'
                f'<span class="dialogue-speaker">NPC</span>{content}</div>'
            )
        else:
            parts.append(
                f'<div class="dialogue-entry dialogue-gm">'
                f'<span class="dialogue-speaker">GM</span>{content}</div>'
            )
    return "".join(parts)


def _parse_voice_choice(choice: str | None) -> str | None:
    """Extract voice_id from 'Name [voice_id]' format, or return None for default."""
    if choice and "[" in choice and choice.endswith("]"):
        return choice.split("[")[-1].rstrip("]").strip()
    return choice or None


def _get_voice_choices() -> list[str]:
    try:
        from tts_service import get_preset_voices
        from voice_store import list_voices
        presets = [f"{v} [preset]" for v in get_preset_voices()]
        cloned  = [f"{v['name']} [{v['voice_id']}]" for v in list_voices()]
        return presets + cloned
    except Exception as e:
        log.warning("Could not load voices: %s", e)
        return ["alba [preset]"]


# ─── Gradio event handlers ────────────────────────────────────────────────────

def refresh_voices() -> list[str]:
    return _get_voice_choices()


def speak_line(text: str, voice_choice: str) -> tuple:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text to speak.")
    voice_id = _parse_voice_choice(voice_choice)
    if not voice_id:
        raise gr.Error("Select a voice first.")
    # Strip '[preset]' suffix for preset voices
    if voice_id == "preset":
        voice_id = voice_choice.split("[")[0].strip()
    try:
        from tts_service import generate as tts_generate
        arr, sr = tts_generate(text, speaker_emb_path=voice_id)
        return (sr, arr)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except RuntimeError as e:
        raise gr.Error(str(e)) from e


def cogm_generate(
    npc_name: str,
    personality: str,
    situation: str,
    history: list,
    voice_choice: str,
) -> tuple:
    npc_name = (npc_name or "").strip()
    personality = (personality or "").strip()
    situation = (situation or "").strip()
    if not npc_name:
        raise gr.Error("Enter an NPC name.")
    if not personality:
        raise gr.Error("Enter NPC personality notes.")
    if not situation:
        raise gr.Error("Describe the current situation.")

    try:
        from ai_service import generate_dialogue
        dialogue = generate_dialogue(npc_name, personality, situation, history)
    except RuntimeError as e:
        raise gr.Error(str(e)) from e

    new_history = (history or []) + [{"role": "assistant", "content": dialogue}]
    new_history = new_history[-20:]

    # TTS for the dialogue
    voice_id = _parse_voice_choice(voice_choice)
    audio_out = None
    if voice_id:
        if voice_id == "preset":
            voice_id = voice_choice.split("[")[0].strip()
        try:
            from tts_service import generate as tts_generate
            arr, sr = tts_generate(dialogue, speaker_emb_path=voice_id)
            audio_out = (sr, arr)
        except Exception as e:
            log.warning("Co-GM TTS failed: %s", e)

    return _render_dialogue_log(new_history), new_history, audio_out


def cogm_clear() -> tuple:
    return '<p style="color:var(--ink-faint);font-size:11px;text-align:center;padding:12px;">Dialogue cleared.</p>', [], None


def hp_changed(a: float, li: float, t: float, z: float, m: float) -> tuple:
    hp = {
        "Aethelred": int(a),
        "Lira":      int(li),
        "Torin":     int(t),
        "Zephyr":    int(z),
        "Mira":      int(m),
    }
    return _render_party_roster(hp), hp


# ─── Gradio Blocks layout ─────────────────────────────────────────────────────

with gr.Blocks(
    css=CSS,
    title="Co-DM Edition",
    theme=gr.themes.Base(
        font=[gr.themes.GoogleFont("IM Fell English"), "Georgia", "serif"],
    ),
    analytics_enabled=False,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── Left column: Quick Tools + Notes ──────────────────
        with gr.Column(scale=1, min_width=220):
            gr.HTML(QUICK_TOOLS_HTML)
            with gr.Group(elem_classes="lb-panel lb-section"):
                gr.HTML('<div class="panel-chrome"><span class="rune">ᛊ</span> SESSION NOTES <span class="rune">ᛊ</span></div>')
                notes_ta = gr.Textbox(
                    lines=8,
                    placeholder="Scribe your notes here…",
                    show_label=False,
                    elem_classes="lb-section",
                )
                with gr.Row():
                    quest_btn  = gr.Button("Quest",  size="sm", elem_classes="lb-btn")
                    danger_btn = gr.Button("Danger", size="sm", elem_classes="lb-btn")
                    lore_btn   = gr.Button("Lore",   size="sm", elem_classes="lb-btn")
                notes_status = gr.HTML("")

        # ── Center column: Banner + Encounter + Spellbook ─────
        with gr.Column(scale=2, min_width=380):
            gr.HTML(CAMPAIGN_BANNER_HTML)
            gr.HTML(ENCOUNTER_TRACKER_HTML)
            gr.HTML(SPELLBOOK_HTML)

        # ── Right column: Party Roster + World Map ────────────
        with gr.Column(scale=1, min_width=240):
            party_state = gr.State(dict(_CHAR_DEFAULT_HP))
            party_html  = gr.HTML(_render_party_roster(_CHAR_DEFAULT_HP))

            with gr.Group(elem_classes="lb-panel lb-section"):
                gr.HTML('<div class="panel-chrome" style="font-size:9px;">UPDATE HP</div>')
                with gr.Row():
                    hp_aeth = gr.Number(value=_CHAR_DEFAULT_HP["Aethelred"], label="Aethelred",
                                        precision=0, minimum=0, maximum=80, elem_classes="lb-section")
                    hp_lira = gr.Number(value=_CHAR_DEFAULT_HP["Lira"], label="Lira",
                                        precision=0, minimum=0, maximum=58, elem_classes="lb-section")
                with gr.Row():
                    hp_tor  = gr.Number(value=_CHAR_DEFAULT_HP["Torin"], label="Torin",
                                        precision=0, minimum=0, maximum=44, elem_classes="lb-section")
                    hp_zeph = gr.Number(value=_CHAR_DEFAULT_HP["Zephyr"], label="Zephyr",
                                        precision=0, minimum=0, maximum=56, elem_classes="lb-section")
                with gr.Row():
                    hp_mira = gr.Number(value=_CHAR_DEFAULT_HP["Mira"], label="Mira",
                                        precision=0, minimum=0, maximum=55, elem_classes="lb-section")

            gr.HTML(WORLD_MAP_HTML)

    # ── TTS + Co-GM panels ─────────────────────────────────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes="lb-panel lb-section"):
            gr.HTML('<div class="panel-chrome"><span class="rune">⚔</span> CHARACTER VOICES <span class="rune">⚔</span></div>')
            with gr.Row():
                voice_dd    = gr.Dropdown(
                    choices=_get_voice_choices(),
                    label="Voice",
                    scale=3,
                    elem_classes="lb-section",
                )
                refresh_btn = gr.Button("↻ Refresh", scale=1, size="sm", elem_classes="lb-btn")
            tts_text = gr.Textbox(
                lines=2, label="Speak a line",
                placeholder="Enter dialogue or narration…",
                elem_classes="lb-section",
            )
            with gr.Row():
                speak_btn = gr.Button("▶ Speak", variant="primary", elem_classes="lb-btn lb-btn-primary")
                tts_audio = gr.Audio(type="numpy", label="Output", autoplay=True)

        with gr.Column(scale=1, elem_classes="lb-panel lb-section"):
            gr.HTML('<div class="panel-chrome"><span class="rune">✦</span> CO-GM ASSISTANT <span class="rune">✦</span></div>')
            with gr.Row():
                npc_name   = gr.Textbox(label="NPC Name", scale=1, elem_classes="lb-section")
                cogm_voice = gr.Dropdown(
                    choices=_get_voice_choices(),
                    label="Voice",
                    scale=1,
                    elem_classes="lb-section",
                )
            personality = gr.Textbox(lines=2, label="Personality / Traits", elem_classes="lb-section")
            situation   = gr.Textbox(lines=2, label="Current Situation",    elem_classes="lb-section")
            with gr.Row():
                gen_btn   = gr.Button("⚔ Speak as NPC", variant="primary", elem_classes="lb-btn lb-btn-primary")
                clear_btn = gr.Button("Clear", variant="secondary",         elem_classes="lb-btn")
            history_state = gr.State([])
            dialogue_log  = gr.HTML(
                '<p style="color:var(--ink-faint);font-size:11px;text-align:center;padding:12px;">No dialogue yet.</p>'
            )
            cogm_audio = gr.Audio(type="numpy", label="NPC Voice", autoplay=True)

    # ── Event wiring ──────────────────────────────────────────
    speak_btn.click(speak_line, [tts_text, voice_dd], tts_audio)
    refresh_btn.click(refresh_voices, [], voice_dd)

    gen_btn.click(
        cogm_generate,
        [npc_name, personality, situation, history_state, cogm_voice],
        [dialogue_log, history_state, cogm_audio],
    )
    clear_btn.click(cogm_clear, [], [dialogue_log, history_state, cogm_audio])

    hp_inputs = [hp_aeth, hp_lira, hp_tor, hp_zeph, hp_mira]
    for hp_inp in hp_inputs:
        hp_inp.change(hp_changed, hp_inputs, [party_html, party_state])

    def _tag_insert(notes, tag):
        return notes + f"\n{tag}: "

    quest_btn.click(_tag_insert,  [notes_ta, gr.State("QUEST")],  notes_ta)
    danger_btn.click(_tag_insert, [notes_ta, gr.State("DANGER")], notes_ta)
    lore_btn.click(_tag_insert,   [notes_ta, gr.State("LORE")],   notes_ta)
