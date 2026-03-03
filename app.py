"""
Gradio UI for Kani TTS. Uses tts_service (thin interface) so the engine can be swapped.
"""
import logging
import os

import gradio as gr

from config import SERVER_NAME

# Default to 7861 so Gradio can run alongside FastAPI (7860)
PORT = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7861")))
from tts_service import generate_to_file

# Pocket TTS: English only; default preset voice
LANG_TAGS = ["en"]
DEFAULT_VOICE = "alba"


def generate(text: str, lang_tag: str):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type something first 🙂")

    try:
        path = generate_to_file(text, language_tag=lang_tag, speaker_emb_path=DEFAULT_VOICE)
        with open(path, "rb") as f:
            # Gradio expects file path; we return path. Returned text from Kani not shown in UI.
            return path, ""
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except RuntimeError as e:
        logging.exception("TTS failed")
        raise gr.Error(str(e)) from e


with gr.Blocks(title="Kani TTS") as demo:
    gr.Markdown("# Kani TTS\nType text, choose a tag, generate audio.")

    text = gr.Textbox(label="Text to speak", lines=4, value="Hello! Kani TTS is working.")
    lang = gr.Dropdown(LANG_TAGS, value="en", label="Language")
    btn = gr.Button("Generate", variant="primary")

    audio_out = gr.Audio(label="Output audio", type="filepath")
    out_text = gr.Textbox(label="Returned text", lines=2)

    btn.click(generate, inputs=[text, lang], outputs=[audio_out, out_text])

demo.launch(server_name=SERVER_NAME, server_port=PORT)
