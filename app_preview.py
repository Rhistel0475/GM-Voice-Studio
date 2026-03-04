"""
Preview Gradio UI for Kani TTS.
Use this file to test style/layout changes before applying them to app.py.
"""
import logging
import os

import gradio as gr

from config import SERVER_NAME
from tts_service import generate_to_file

PORT = int(os.environ.get("GRADIO_PREVIEW_PORT", "7863"))

LANG_TAGS = ["en"]
DEFAULT_VOICE = "alba"

PREVIEW_CSS = """
:root {
  --bg: #0f0d11;
  --shell: #171a24;
  --panel: #211a19;
  --panel-2: #2b2220;
  --gold: #b3905f;
  --gold-2: #d9b375;
  --text: #f2e4c9;
  --muted: #ccb792;
  --accent: #4ec0d6;
}

.gradio-container {
  background:
    radial-gradient(circle at 50% -20%, rgba(77, 166, 199, 0.18), transparent 46%),
    radial-gradient(circle at 12% 24%, rgba(231, 179, 87, 0.12), transparent 32%),
    radial-gradient(circle at 87% 74%, rgba(96, 57, 32, 0.26), transparent 42%),
    var(--bg);
  color: var(--text);
  font-family: "Crimson Text", Georgia, "Times New Roman", serif;
  padding: 12px !important;
}

.gradio-container .main {
  max-width: 1020px;
  margin: 0 auto;
  border-radius: 16px;
  border: 3px solid #111521;
  box-shadow:
    0 0 0 1px rgba(67, 52, 35, 0.8),
    0 0 0 2px rgba(72, 179, 197, 0.28),
    0 24px 65px rgba(0, 0, 0, 0.7);
  background: linear-gradient(180deg, #1f2230, #161826 12%, #14141a 12%, #14141a 100%);
  padding: 12px 14px 14px !important;
  position: relative;
}

.gradio-container .main::after {
  content: "ᚱ ᛉ ᚠ";
  position: absolute;
  top: 10px;
  right: 14px;
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  color: rgba(106, 210, 225, 0.8);
}

.codm-title {
  margin: 2px 2px 10px;
  text-align: center;
  color: var(--gold-2);
  font-family: "Cinzel", Georgia, serif;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-shadow: 0 0 18px rgba(217, 179, 117, 0.35);
}

.codm-sub {
  text-align: center;
  color: #7fd7e3;
  letter-spacing: 0.1em;
  font-size: 0.78rem;
  margin-bottom: 10px;
}

.gradio-container .panel {
  border: 1px solid #6a5137;
  border-radius: 9px;
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  box-shadow: inset 0 0 0 1px #3c2f24;
  padding: 9px;
  margin-bottom: 8px;
}

.gradio-container .panel:hover {
  border-color: #8a6a45;
  box-shadow: inset 0 0 0 1px #4a3a2d, 0 0 10px rgba(78, 192, 214, 0.15);
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container p {
  color: var(--text) !important;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(180deg, #7d623f, #5f492f) !important;
  border: 1px solid var(--gold) !important;
  color: #f8efd9 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  min-height: 40px;
}

.gradio-container .gr-button-primary:hover {
  filter: brightness(1.08);
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  background: #1b1514 !important;
  border: 1px solid #5e4a34 !important;
  color: var(--text) !important;
  border-radius: 7px !important;
}

.gradio-container .panel .form,
.gradio-container .panel .wrap {
  background: transparent !important;
  border: none !important;
}

.gradio-container .panel-head {
  font-family: "Cinzel", Georgia, serif;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #f0dbb5;
  border-bottom: 1px solid rgba(87, 146, 160, 0.45);
  margin-bottom: 7px;
  padding-bottom: 3px;
}

.dock-strip {
  margin-top: 6px;
  border-top: 1px solid rgba(84, 178, 197, 0.3);
  padding-top: 8px;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 5px;
}

.dock-item {
  border: 1px solid rgba(92, 74, 48, 0.75);
  border-radius: 5px;
  background: linear-gradient(180deg, rgba(74, 57, 35, 0.45), rgba(22, 18, 12, 0.88));
  text-align: center;
  padding: 5px 4px;
  font-family: "Cinzel", Georgia, serif;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.54rem;
  color: #d3bf97;
}

.dock-item span {
  display: block;
  color: #6ed2e0;
  margin-bottom: 2px;
}
"""


def generate(text: str, lang_tag: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type something first 🙂")

    try:
        path = generate_to_file(text, language_tag=lang_tag, speaker_emb_path=DEFAULT_VOICE)
        return path, ""
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except RuntimeError as e:
        logging.exception("TTS failed")
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Kani TTS - Preview", css=PREVIEW_CSS, theme=gr.themes.Base()) as demo:
  gr.Markdown("""
<div class="codm-title">CO-DM Voice Studio Preview</div>
<div class="codm-sub">Dungeon Master's Companion • Styling Sandbox</div>
""")

  with gr.Row():
    with gr.Column(scale=3, elem_classes=["panel"]):
      gr.Markdown('<div class="panel-head">Narration Console</div>')
      text = gr.Textbox(label="Text to speak", lines=6, value="Hello! Kani TTS is working.")
      btn = gr.Button("Generate", variant="primary")
      audio_out = gr.Audio(label="Output audio", type="filepath")

    with gr.Column(scale=2, elem_classes=["panel"]):
      gr.Markdown('<div class="panel-head">Voice Controls</div>')
      lang = gr.Dropdown(LANG_TAGS, value="en", label="Language")
      out_text = gr.Textbox(label="Returned text", lines=4)

  btn.click(generate, inputs=[text, lang], outputs=[audio_out, out_text])

  gr.Markdown("""
<div class="dock-strip">
  <div class="dock-item"><span>◆</span>Dashboard</div>
  <div class="dock-item"><span>⚑</span>Campaigns</div>
  <div class="dock-item"><span>📜</span>Library</div>
  <div class="dock-item"><span>⚒</span>Tools</div>
  <div class="dock-item"><span>⚙</span>Settings</div>
</div>
""")


demo.launch(server_name=SERVER_NAME, server_port=PORT)
