#!/usr/bin/env python3
"""
Minimal standalone TTS reproduction for Kani TTS dependency/behavior testing.
Run outside the FastAPI app to isolate kani_tts + transformers + torch.

Usage:
  .venv/bin/python scripts/tts_repro.py                    # no speaker emb
  .venv/bin/python scripts/tts_repro.py path/to/voice.pt  # with speaker emb (repro RoPE mismatch)

Prints env (torch, transformers, kani), runs one short generation, then exits.
Applies same compatibility patches as app (Lfm2Config.rope_theta, TransformersKwargs/Unpack).
"""
import sys

def _apply_app_compat_patches():
    """Mirror patches from app/main.py and tts_service so repro runs without starting the app."""
    from typing import TypedDict, Unpack
    import transformers.utils as tf_utils
    import transformers.processing_utils as tf_proc
    if not hasattr(tf_utils, "TransformersKwargs"):
        tf_utils.TransformersKwargs = TypedDict("TransformersKwargs", {}, total=False)
    if not hasattr(tf_proc, "Unpack"):
        tf_proc.Unpack = Unpack
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
    if not hasattr(Lfm2Config, "rope_theta"):
        @property
        def _rope_theta_prop(self):
            rp = getattr(self, "rope_parameters", None)
            if rp is not None and isinstance(rp, dict):
                return rp.get("rope_theta", 10000.0)
            if rp is not None and hasattr(rp, "rope_theta"):
                return rp.rope_theta
            return 10000.0
        Lfm2Config.rope_theta = _rope_theta_prop
    # kani_tts uses _tied_weights_keys = ["lm_head.weight"] (list); transformers 5.x expects dict
    from transformers.modeling_utils import PreTrainedModel
    if not getattr(PreTrainedModel, "_tts_repro_tied_patched", False):
        _orig_tied = PreTrainedModel.get_expanded_tied_weights_keys
        def _patched_tied(self, all_submodels=False):
            keys = getattr(self, "_tied_weights_keys", None)
            if isinstance(keys, list):
                self._tied_weights_keys = {k: "model.embed_tokens.weight" for k in keys}
            try:
                return _orig_tied(self, all_submodels)
            finally:
                if isinstance(keys, list):
                    self._tied_weights_keys = keys
        PreTrainedModel.get_expanded_tied_weights_keys = _patched_tied
        PreTrainedModel._tts_repro_tied_patched = True
    # kani_tts forward uses pos_emb; transformers 5 Lfm2Model has rotary_emb
    from kani_tts.model import Lfm2ForKaniModel
    if not getattr(Lfm2ForKaniModel, "_pos_emb_patched", False):
        _orig_init = Lfm2ForKaniModel.__init__
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.pos_emb = getattr(self, "rotary_emb", None)
        Lfm2ForKaniModel.__init__ = _patched_init
        Lfm2ForKaniModel._pos_emb_patched = True

def main():
    print("Environment:")
    import torch
    import transformers
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    try:
        from kani_tts import __version__ as kani_version
        print(f"  kani_tts (module): {kani_version}")
    except Exception as e:
        print(f"  kani_tts: import error - {e}")
        return 1
    try:
        from importlib.metadata import version
        print(f"  kani-tts-2 (package): {version('kani-tts-2')}")
    except Exception:
        pass

    speaker_emb_path = sys.argv[1].strip() if len(sys.argv) > 1 else None
    speaker_emb = None
    if speaker_emb_path:
        import os
        if not os.path.isfile(speaker_emb_path):
            print(f"Error: file not found: {speaker_emb_path}")
            return 1
        loaded = torch.load(speaker_emb_path, weights_only=True)
        if isinstance(loaded, torch.Tensor):
            speaker_emb = loaded
        elif isinstance(loaded, list) and loaded and isinstance(loaded[0], torch.Tensor):
            speaker_emb = loaded[0]
        elif isinstance(loaded, dict):
            vals = [v for v in loaded.values() if isinstance(v, torch.Tensor) and v.dim() in (1, 2)]
            if vals:
                speaker_emb = vals[0]
        if speaker_emb is None:
            print("Error: no tensor found in .pt file")
            return 1
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        print(f"Speaker emb: {speaker_emb.shape}")
    else:
        print("No speaker emb (default voice).")

    _apply_app_compat_patches()
    print("Loading KaniTTS...")
    from kani_tts import KaniTTS
    model = KaniTTS("nineninesix/kani-tts-2-en", max_new_tokens=500)
    causal_lm = getattr(getattr(model, "model", None), "model", None)
    if causal_lm is not None and hasattr(causal_lm, "set_attn_implementation"):
        causal_lm.set_attn_implementation("eager")
        print("Attention backend: eager")
    print("Generating...")
    text = "Hello, this is a short test."
    try:
        kwargs = dict(language_tag="en_us", temperature=0.9, top_p=0.95, repetition_penalty=1.1)
        if speaker_emb is not None:
            audio, out_text = model(text, speaker_emb=speaker_emb, **kwargs)
        else:
            audio, out_text = model(text, **kwargs)
        import numpy as np
        arr = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
        print(f"OK: audio shape={arr.shape}, sr=22050")
        return 0
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
