"""
TTS service: thin interface over KaniTTS-2 (nineninesix.ai).
Callers get (audio_array, sample_rate). English-only; supports cloned voices (.pt tensors).
No preset named voices — pass speaker_emb_path=None for a random voice.
"""
import functools
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import soundfile as sf
import torch

from app.core.config import (
    AUDIO_CACHE_SIZE,
    DEFAULT_VOICE_ID,
    HUME_API_KEY,
    HUME_BASE_URL,
    HUME_TTS_VERSION,
    TTS_PROVIDER,
)

# --- Compatibility shim for kani_tts.model (requires TransformersKwargs / Unpack from newer transformers) ---
def _patch_transformers_for_kani_tts():
    import sys
    if "kani_tts" in sys.modules:
        return
    try:
        import transformers.utils as tf_utils
        if not hasattr(tf_utils, "TransformersKwargs"):
            tf_utils.TransformersKwargs = TypedDict("TransformersKwargs", {}, total=False)
    except Exception:
        pass
    try:
        from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
        if not hasattr(Lfm2Config, "rope_theta"):
            @property
            def _rope_theta_prop(self):
                rope_parameters = getattr(self, "rope_parameters", None)
                if isinstance(rope_parameters, dict):
                    return rope_parameters.get("rope_theta", 10000.0)
                if rope_parameters is not None and hasattr(rope_parameters, "rope_theta"):
                    return rope_parameters.rope_theta
                return 10000.0

            Lfm2Config.rope_theta = _rope_theta_prop
    except Exception:
        pass
    try:
        from transformers.modeling_utils import PreTrainedModel
        if not getattr(PreTrainedModel, "_kani_tts_tied_weights_patched", False):
            _orig_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys

            def _patched_get_expanded(self, all_submodels=False):
                keys = getattr(self, "_tied_weights_keys", None)
                if isinstance(keys, list):
                    self._tied_weights_keys = {k: "model.embed_tokens.weight" for k in keys}
                try:
                    return _orig_get_expanded(self, all_submodels)
                finally:
                    if isinstance(keys, list):
                        self._tied_weights_keys = keys

            PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded
            PreTrainedModel._kani_tts_tied_weights_patched = True
    except Exception:
        pass
    try:
        import transformers.processing_utils as tf_proc
        if not hasattr(tf_proc, "Unpack"):
            from typing import Unpack as _Unpack
            tf_proc.Unpack = _Unpack
    except Exception:
        pass


_patch_transformers_for_kani_tts()

KANI_SAMPLE_RATE = 22050
DEFAULT_TTS_TEMPERATURE = 0.65
DEFAULT_TTS_TOP_P = 0.80
DEFAULT_TTS_REPETITION_PENALTY = 1.15
HUME_SAMPLE_RATE = 24000
HUME_VOICE_ID_PREFIX = "hume"
HUME_PROVIDER_HUME_AI = "HUME_AI"
HUME_PROVIDER_CUSTOM_VOICE = "CUSTOM_VOICE"

# KaniTTS-2 English variants exposed by the model.
DEFAULT_LANGUAGE_TAG = "en_us"
SUPPORTED_LANGUAGE_TAGS = [
    "en_us",
    "en_nyork",
    "en_oakl",
    "en_glasg",
    "en_bost",
    "en_scou",
]

_model = None
_audio_cache: list[str] = []


def get_tts_provider() -> str:
    return TTS_PROVIDER


def is_hume_provider() -> bool:
    return get_tts_provider() == "hume"


def make_hume_voice_id(provider: str, voice_id: str) -> str:
    return f"{HUME_VOICE_ID_PREFIX}:{provider}:{voice_id}"


def parse_hume_voice_id(voice_id: Optional[str]) -> Optional[tuple[str, str]]:
    raw = (voice_id or "").strip()
    if not raw.startswith(f"{HUME_VOICE_ID_PREFIX}:"):
        return None
    parts = raw.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def is_hume_voice_id(voice_id: Optional[str]) -> bool:
    return parse_hume_voice_id(voice_id) is not None


def resolve_hume_voice_id(voice_id: Optional[str]) -> Optional[str]:
    raw = (voice_id or "").strip()
    if not raw:
        return None
    if is_hume_voice_id(raw):
        return raw

    matches = [voice for voice in list_hume_voices() if (voice.get("provider_voice_id") or "").strip() == raw]
    if len(matches) == 1:
        return matches[0]["voice_id"]
    return None


def _hume_headers() -> dict[str, str]:
    if not HUME_API_KEY:
        raise RuntimeError("Hume API key not configured. Set HUME_API_KEY.")
    return {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "GM-Voice-Studio/1.0 (+https://api.hume.ai)",
    }


def _hume_request(path: str, payload: Optional[dict] = None, method: str = "GET") -> bytes:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{HUME_BASE_URL.rstrip('/')}{path}",
        data=data,
        method=method,
        headers=_hume_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hume API request failed ({exc.code}): {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Hume API: {exc.reason}") from exc


def list_hume_voices() -> list[dict]:
    if not HUME_API_KEY:
        logging.warning("Hume voice list requested but HUME_API_KEY is not configured.")
        return []

    voices: list[dict] = []
    try:
        for provider in (HUME_PROVIDER_HUME_AI, HUME_PROVIDER_CUSTOM_VOICE):
            raw = _hume_request(f"/v0/tts/voices?provider={provider}")
            payload = json.loads(raw.decode("utf-8"))
            items = payload.get("voices_page") or payload.get("voices") or payload.get("items") or []
            logging.info("Fetched %d Hume voices for provider=%s", len(items), provider)
            for item in items:
                remote_id = str(item.get("id") or item.get("voice_id") or "").strip()
                if not remote_id:
                    continue
                is_default = bool(DEFAULT_VOICE_ID and remote_id == DEFAULT_VOICE_ID.strip())
                name = (
                    item.get("name")
                    or item.get("display_name")
                    or item.get("description")
                    or remote_id
                )
                description = item.get("description") or ""
                tags = []
                if is_default:
                    name = "Master Voice"
                    description = description or "Featured Hume voice for primary narration and character performance."
                    tags = ["featured", "master-voice"]
                voices.append({
                    "voice_id": make_hume_voice_id(provider, remote_id),
                    "provider_voice_id": remote_id,
                    "provider": provider,
                    "name": str(name).strip() or remote_id,
                    "source": "system" if provider == HUME_PROVIDER_HUME_AI else "custom",
                    "status": "ready",
                    "description": description,
                    "tags": tags,
                    "icon": "crown" if is_default else None,
                    "featured": is_default,
                })
    except Exception as exc:
        logging.warning("Failed to fetch Hume voices: %s", exc)
    preferred_id = None
    if DEFAULT_VOICE_ID:
        raw_preferred = DEFAULT_VOICE_ID.strip()
        for voice in voices:
            if raw_preferred in {
                (voice.get("voice_id") or "").strip(),
                (voice.get("provider_voice_id") or "").strip(),
            }:
                preferred_id = voice.get("voice_id")
                break
        if preferred_id is None:
            synthetic_voice_id = make_hume_voice_id(HUME_PROVIDER_HUME_AI, raw_preferred)
            voices.append({
                "voice_id": synthetic_voice_id,
                "provider_voice_id": raw_preferred,
                "provider": HUME_PROVIDER_HUME_AI,
                "name": "Master Voice",
                "source": "system",
                "status": "ready",
                "description": "Featured Hume voice configured locally. If playback fails, verify this voice ID exists in your Hume account or Voice Library access.",
                "tags": ["featured", "master-voice"],
                "icon": "crown",
                "featured": True,
            })
            preferred_id = synthetic_voice_id
    if preferred_id:
        voices.sort(key=lambda voice: 0 if voice.get("voice_id") == preferred_id else 1)
    logging.info("Resolved %d total Hume voices", len(voices))
    return voices


def get_hume_voice(voice_id: str) -> Optional[dict]:
    resolved = resolve_hume_voice_id(voice_id)
    wanted = parse_hume_voice_id(resolved)
    if not wanted:
        return None
    provider, remote_id = wanted
    for voice in list_hume_voices():
        current = parse_hume_voice_id(voice.get("voice_id"))
        if current == (provider, remote_id):
            return voice
    return None


def delete_hume_voice(voice_id: str) -> bool:
    parsed = parse_hume_voice_id(resolve_hume_voice_id(voice_id))
    if not parsed:
        return False
    provider, remote_id = parsed
    if provider != HUME_PROVIDER_CUSTOM_VOICE:
        raise ValueError("Built-in Hume voices cannot be deleted.")
    _hume_request(f"/v0/tts/voices/{remote_id}", method="DELETE")
    return True


def _generate_hume(
    text: str,
    speaker_emb_path: Optional[str],
) -> tuple[np.ndarray, int]:
    voice_payload = None
    resolved_voice_id = resolve_hume_voice_id(speaker_emb_path)
    if not resolved_voice_id and DEFAULT_VOICE_ID:
        resolved_voice_id = resolve_hume_voice_id(DEFAULT_VOICE_ID)

    parsed = parse_hume_voice_id(resolved_voice_id)
    if parsed:
        provider, remote_id = parsed
        voice_payload = {
            "id": remote_id,
        }
        logging.info("Generating Hume TTS with voice_id=%s provider=%s", remote_id, provider)
    else:
        logging.info("Generating Hume TTS without explicit voice_id")

    payload: dict = {
        "utterances": [
            {
                "text": text,
            }
        ],
        "format": {"type": "wav"},
        "version": HUME_TTS_VERSION,
    }
    if voice_payload is not None:
        payload["utterances"][0]["voice"] = voice_payload
    elif HUME_TTS_VERSION == "2":
        raise ValueError("Hume Octave 2 requires a voice. Check DEFAULT_VOICE_ID or select a voice from /voices/list.")

    audio_bytes = _hume_request("/v0/tts/file", payload=payload, method="POST")
    logging.info("Received %d bytes from Hume TTS", len(audio_bytes))
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate or HUME_SAMPLE_RATE)


def _install_local_kani_tts_model():
    """
    Force kani_tts.core to import the repo's fixed model module.

    The packaged kani_tts runtime pulls in kani_tts.model from site-packages,
    but this repo carries a newer frame-position-aware implementation that is
    required for stable KaniTTS-2 inference.
    """
    local_model_path = Path(__file__).resolve().parents[2] / "kani_tts_model_fixed.py"
    if not local_model_path.is_file():
        logging.warning(
            "Local KaniTTS model override not found at %s; using package model.",
            local_model_path,
        )
        return

    existing = sys.modules.get("kani_tts.model")
    existing_path = Path(getattr(existing, "__file__", "")) if existing is not None else None
    if existing_path and existing_path.resolve() == local_model_path.resolve():
        return

    for module_name in ("kani_tts.api", "kani_tts.core", "kani_tts.model", "kani_tts"):
        sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location("kani_tts.model", str(local_model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local KaniTTS model override from {local_model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["kani_tts.model"] = module
    spec.loader.exec_module(module)
    logging.info("Installed local KaniTTS model override from %s", local_model_path)


def _get_tts():
    global _model
    if _model is None:
        try:
            _install_local_kani_tts_model()
            # transformers 5.x Lfm2Model uses rotary_emb; kani_tts forward expects pos_emb
            from kani_tts.model import Lfm2ForKaniModel
            if not hasattr(Lfm2ForKaniModel, "_pos_emb_patched"):
                _orig_init = Lfm2ForKaniModel.__init__

                def _patched_init(self, *args, **kwargs):
                    _orig_init(self, *args, **kwargs)
                    self.pos_emb = getattr(self, "rotary_emb", None)
                    if self.pos_emb is None:
                        raise RuntimeError("Lfm2ForKaniModel: rotary_emb not set by parent Lfm2Model")

                Lfm2ForKaniModel.__init__ = _patched_init
                Lfm2ForKaniModel._pos_emb_patched = True

            from kani_tts import KaniTTS
            from kani_tts.model import KaniTTS2ForCausalLM
            # Fix N vs N-1 tensor size mismatch in apply_rotary_pos_emb when a speaker
            # embedding is inserted (seq length N → N+1 but position_ids/cache_position
            # pre-computed by GenerationMixin.generate() from the original attention_mask of
            # length N).  Force all three to match inputs_embeds at two choke points:
            #   Patch 2 – prepare_inputs_for_generation: fix returned dict before forward.
            #   Patch 3 – KaniTTS2ForCausalLM.forward: safety-net + debug log before inner call.
            if not getattr(KaniTTS2ForCausalLM, "_position_sync_patched", False):
                _orig_prepare = KaniTTS2ForCausalLM.prepare_inputs_for_generation
                _orig_forward = KaniTTS2ForCausalLM.forward

                @functools.wraps(_orig_prepare)
                def _patched_prepare(self, *args, **kwargs):
                    out = _orig_prepare(self, *args, **kwargs)
                    emb = out.get("inputs_embeds")
                    if emb is not None:
                        seq_len = emb.shape[1]
                        dev = emb.device
                        cp = out.get("cache_position")
                        if cp is None or cp.shape[0] != seq_len:
                            out["cache_position"] = torch.arange(seq_len, device=dev, dtype=torch.long)
                            logging.debug(
                                "[TTS patch-prepare] fixed cache_position %s → arange(%d)",
                                getattr(cp, "shape", None), seq_len,
                            )
                        pos = out.get("position_ids")
                        if pos is None or pos.dim() < 2 or pos.shape[-1] != seq_len:
                            out["position_ids"] = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
                            logging.debug(
                                "[TTS patch-prepare] fixed position_ids %s → [1, %d]",
                                getattr(pos, "shape", None), seq_len,
                            )
                    return out

                @functools.wraps(_orig_forward)
                def _patched_forward(self, *args, **kwargs):
                    inputs_embeds = kwargs.get("inputs_embeds")
                    if inputs_embeds is not None:
                        seq_len = inputs_embeds.shape[1]
                        dev = inputs_embeds.device
                        cp = kwargs.get("cache_position")
                        pos = kwargs.get("position_ids")
                        logging.debug(
                            "[TTS patch-fwd] prefill: emb=%s pos_ids=%s cache_pos=%s",
                            tuple(inputs_embeds.shape),
                            getattr(pos, "shape", None),
                            getattr(cp, "shape", None),
                        )
                        if cp is None or cp.shape[0] != seq_len:
                            kwargs["cache_position"] = torch.arange(seq_len, device=dev, dtype=torch.long)
                        if pos is None or pos.dim() < 2 or pos.shape[-1] != seq_len:
                            kwargs["position_ids"] = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
                    return _orig_forward(self, *args, **kwargs)

                KaniTTS2ForCausalLM.prepare_inputs_for_generation = _patched_prepare
                KaniTTS2ForCausalLM.forward = _patched_forward
                KaniTTS2ForCausalLM._position_sync_patched = True

            # Patch 4 – Lfm2ForKaniModel.forward: innermost safeguard right before the
            # super().forward() (Lfm2Model) call that builds RoPE from position_ids.
            # If inputs_embeds is present and cache_position / position_ids have the wrong
            # length, fix them here so RoPE cos/sin match query/key sequence length.
            if not getattr(Lfm2ForKaniModel, "_rope_sync_patched", False):
                _orig_kani_fwd = Lfm2ForKaniModel.forward

                @functools.wraps(_orig_kani_fwd)
                def _patched_kani_fwd(self, *args, **kwargs):
                    inputs_embeds = kwargs.get("inputs_embeds")
                    if inputs_embeds is not None and not getattr(self, "use_learnable_rope", True):
                        seq_len = inputs_embeds.shape[1]
                        dev = inputs_embeds.device
                        cp = kwargs.get("cache_position")
                        pos = kwargs.get("position_ids")
                        if cp is None or cp.shape[0] != seq_len:
                            logging.debug(
                                "[TTS inner-patch] Lfm2ForKaniModel: cache_position %s → arange(%d)",
                                getattr(cp, "shape", None), seq_len,
                            )
                            kwargs["cache_position"] = torch.arange(seq_len, device=dev, dtype=torch.long)
                        if pos is None or pos.dim() < 2 or pos.shape[-1] != seq_len:
                            logging.debug(
                                "[TTS inner-patch] Lfm2ForKaniModel: position_ids %s → [1, %d]",
                                getattr(pos, "shape", None), seq_len,
                            )
                            kwargs["position_ids"] = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
                    return _orig_kani_fwd(self, *args, **kwargs)

                Lfm2ForKaniModel.forward = _patched_kani_fwd
                Lfm2ForKaniModel._rope_sync_patched = True

            # Patch 5 – intercept Lfm2Attention.forward to fix causal mask shape mismatch.
            # When the causal mask k_len (mask.shape[-1]) doesn't match the actual key k_len,
            # discard the mask (fall back to is_causal=True in SDPA) rather than crashing.
            from transformers.models.lfm2.modeling_lfm2 import (
                Lfm2Attention,
                ALL_ATTENTION_FUNCTIONS,
                apply_rotary_pos_emb,
                eager_attention_forward,
            )
            if not getattr(Lfm2Attention, "_mask_compat_patched", False):
                @functools.wraps(Lfm2Attention.forward)
                def _patched_attn_fwd(self, hidden_states, position_embeddings,
                                      attention_mask=None, past_key_values=None,
                                      cache_position=None, **kwargs):
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)

                    query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
                    key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
                    value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                    if past_key_values is not None:
                        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                        key_states, value_states = past_key_values.update(
                            key_states, value_states, self.layer_idx, cache_kwargs
                        )
                        if key_states.shape[-2] != value_states.shape[-2]:
                            compatible_k_len = min(key_states.shape[-2], value_states.shape[-2])
                            logging.debug(
                                "[TTS patch-attn] cropping kv lengths from k=%d v=%d to %d",
                                key_states.shape[-2], value_states.shape[-2], compatible_k_len,
                            )
                            key_states = key_states[..., :compatible_k_len, :]
                            value_states = value_states[..., :compatible_k_len, :]

                    if attention_mask is not None and attention_mask.dim() == 4:
                        q_len = query_states.shape[-2]
                        k_len = key_states.shape[-2]
                        if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != k_len:
                            logging.debug(
                                "[TTS patch-attn] cropping mask from %s to (..., %d, %d)",
                                tuple(attention_mask.shape), q_len, k_len,
                            )
                            attention_mask = attention_mask[..., :q_len, :k_len]
                        if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != k_len:
                            logging.debug(
                                "[TTS patch-attn] mask still incompatible after crop (%s), dropping mask",
                                tuple(attention_mask.shape),
                            )
                            attention_mask = None

                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        self.config._attn_implementation, eager_attention_forward
                    )
                    attn_output, attn_weights = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0,
                        scaling=self.scaling,
                        **kwargs,
                    )
                    batch_size = hidden_states.shape[0] if hidden_states.dim() >= 3 else 1
                    seq_len = query_states.shape[-2]
                    attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
                    output = self.out_proj(attn_output)
                    return output, attn_weights

                Lfm2Attention.forward = _patched_attn_fwd
                Lfm2Attention._mask_compat_patched = True

            logging.info("Loading KaniTTS-2...")
            _model = KaniTTS(
                "nineninesix/kani-tts-2-en",
                max_new_tokens=6000,
                show_info=False,
                # The published config enables learnable RoPE, but the current
                # inference path is more stable with standard RoPE.
                use_learnable_rope=False,
            )
            causal_lm = getattr(getattr(_model, "model", None), "model", None)
            if getattr(causal_lm, "use_learnable_rope", None):
                logging.warning("KaniTTS-2 inference still has learnable RoPE enabled after override.")
            else:
                logging.info("Disabled learnable RoPE for KaniTTS-2 inference.")
            if causal_lm is not None and hasattr(causal_lm, "set_attn_implementation"):
                causal_lm.set_attn_implementation("eager")
                logging.info("Forced KaniTTS-2 attention backend to eager for transformers 5.x compatibility.")
            logging.info("KaniTTS-2 loaded.")
        except Exception as e:
            logging.exception("Failed to load KaniTTS-2: %s", e)
            raise
    return _model


def is_model_loaded() -> bool:
    if is_hume_provider():
        return bool(HUME_API_KEY)
    return _model is not None


def get_supported_language_tags() -> list[str]:
    if is_hume_provider():
        return ["en"]
    return list(SUPPORTED_LANGUAGE_TAGS)


def get_preset_voices() -> list[str]:
    """KaniTTS-2 has no named preset voices; returns empty list."""
    if is_hume_provider():
        return [voice["voice_id"] for voice in list_hume_voices()]
    return []


def _is_preset_voice(voice_id: str) -> bool:
    """KaniTTS-2 has no named preset voices; always returns False."""
    if is_hume_provider():
        return resolve_hume_voice_id(voice_id) is not None
    return False


def _normalize_language_tag(language_tag: Optional[str]) -> str:
    """Map legacy/simplified tags to a valid KaniTTS-2 language tag."""
    tag = (language_tag or "").strip().lower()
    if not tag or tag == "en":
        return DEFAULT_LANGUAGE_TAG
    if tag not in SUPPORTED_LANGUAGE_TAGS:
        raise ValueError(
            "Unsupported language_tag. Use one of: " + ", ".join(SUPPORTED_LANGUAGE_TAGS)
        )
    return tag


def _evict_old_audio():
    while len(_audio_cache) >= AUDIO_CACHE_SIZE and _audio_cache:
        path = _audio_cache.pop(0)
        try:
            os.unlink(path)
        except OSError:
            pass


def generate(
    text: str,
    language_tag: Optional[str] = DEFAULT_LANGUAGE_TAG,
    speaker_emb_path: Optional[str] = None,
    temperature: float = DEFAULT_TTS_TEMPERATURE,
    top_p: float = DEFAULT_TTS_TOP_P,
    repetition_penalty: float = DEFAULT_TTS_REPETITION_PENALTY,
) -> tuple[np.ndarray, int]:
    """Generate speech.
    speaker_emb_path: path to a .pt speaker embedding file produced by clone_voice().
    Pass None for a random/default voice.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")
    if is_hume_provider():
        return _generate_hume(text=text, speaker_emb_path=speaker_emb_path)
    normalized_language_tag = _normalize_language_tag(language_tag)

    if speaker_emb_path and speaker_emb_path.strip():
        p = Path(speaker_emb_path.strip())
        if not p.exists():
            raise ValueError("Voice not found. Select a cloned voice or leave voice unset for random.")
        loaded = torch.load(str(p), weights_only=True)
        # Normalize to tensor: .pt may be a single tensor, or list of tensors (e.g. from older save)
        if isinstance(loaded, list):
            if len(loaded) == 0:
                raise ValueError("Voice file is empty")
            emb = loaded[0] if isinstance(loaded[0], torch.Tensor) else torch.tensor(loaded[0], dtype=torch.float32)
        elif isinstance(loaded, torch.Tensor):
            emb = loaded
        elif isinstance(loaded, dict):
            # state_dict or single-key dict: use first value that looks like an embedding
            vals = [v for v in loaded.values() if isinstance(v, torch.Tensor) and v.dim() in (1, 2)]
            if not vals:
                raise ValueError("Voice file has no tensor embedding")
            emb = vals[0]
        else:
            raise ValueError(f"Voice file format not supported: {type(loaded)}")
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
    else:
        emb = None

    model = _get_tts()
    try:
        kwargs = dict(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
        if emb is not None:
            audio, _out_text = model(
                text,
                language_tag=normalized_language_tag,
                speaker_emb=emb,
                **kwargs,
            )
        else:
            audio, _out_text = model(text, language_tag=normalized_language_tag, **kwargs)
        if hasattr(audio, "cpu"):
            arr = audio.cpu().numpy()
        elif isinstance(audio, np.ndarray):
            arr = audio
        else:
            arr = np.array(audio)
    except Exception as e:
        logging.exception("TTS generate failed")
        # Graceful fallback: return 1s silence so narration/text flow can still complete
        fallback_seconds = 1.0
        silence = np.zeros(int(KANI_SAMPLE_RATE * fallback_seconds), dtype=np.float32)
        logging.warning("Returning %.1fs silence due to TTS failure; narration continues.", fallback_seconds)
        return silence, KANI_SAMPLE_RATE

    return arr, KANI_SAMPLE_RATE


def generate_to_file(
    text: str,
    language_tag: Optional[str] = DEFAULT_LANGUAGE_TAG,
    speaker_emb_path: Optional[str] = None,
) -> str:
    audio, sample_rate = generate(text, language_tag=language_tag, speaker_emb_path=speaker_emb_path)
    _evict_old_audio()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, sample_rate)
    except Exception as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(f"Could not save audio: {e!s}") from e
    _audio_cache.append(path)
    return path
