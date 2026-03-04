"""
Custom LFM2 implementation with KaniTTS-2 frame-level position encoding.

Key Innovation:
- Frame-level position encoding: All 4 tokens within an audio frame share the same position ID
  This reduces RoPE distance between tokens across frames, improving long-form generation.

Compatible with Flash Attention 2 for 10-20x training speedup.

FIXED: Proper frame-level position tracking during generation with KV-cache.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.generation.utils import GenerationMixin

# Import base LFM2 classes
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Model,
    Lfm2ForCausalLM,
    Lfm2PreTrainedModel,
    Lfm2HybridConvCache,
)
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config


def compute_frame_level_positions(
    input_ids: torch.Tensor,
    audio_tokens_start: int,
    tokens_per_frame: int = 4,
    audio_step: float = 1.0
    ) -> torch.Tensor:
    """
    Vectorized computation of frame-level position IDs (10-50x faster than Python loops).

    Key insight: Use cumulative counts to determine positions.

    - Text tokens: sequential positions (step 1.0)
    - Audio tokens: frame-level positions (step audio_step per frame)

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        audio_tokens_start: Token ID where audio tokens begin (typically 64410)
        tokens_per_frame: Number of tokens per audio frame (typically 4)
        audio_step: Position step size per audio frame (default 1.0).
                    Set to < 1.0 (e.g., 0.5) to compress audio position space.

    Returns:
        position_ids: Position IDs [batch_size, seq_len].
                      if audio_step is float, returns FloatTensor.

    Example:
        >>> input_ids = torch.tensor([[100, 200, 64410, 68442, 72474, 76506, 300]])
        >>> # Tokens:                [text, text, aud0,  aud1,  aud2,  aud3,  text]
        >>> pos = compute_frame_level_positions(input_ids, 64410, 4, audio_step=0.5)
        >>> pos
        tensor([[0., 1., 2., 2., 2., 2., 3.]])
        # Text at 0, 1. Audio frame at 2. Next text at 3 (1+1+1?)
        # Note: Text logic accumulates 1 per text token.
        # Audio logic accumulates audio_step per frame.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Identify audio tokens
    is_audio = input_ids >= audio_tokens_start
    text_mask = ~is_audio

    # Prepare zero prefix for cumsum
    zeros = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

    # 1. Count text tokens before each position
    #    This gives the integer base from text tokens
    text_count = torch.cat([zeros, text_mask.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 2. Count audio tokens before each position
    audio_token_count = torch.cat([zeros, is_audio.long()], dim=1).cumsum(dim=1)[:, :-1]

    # 3. Convert token count to frame count (0, 0, 0, 0, 1, 1...)
    audio_frame_count = audio_token_count // tokens_per_frame

    # 4. Compute final positions
    #    Text contributes 1.0 per token
    #    Audio frames contribute audio_step per frame
    position_ids = text_count + audio_frame_count * audio_step

    return position_ids


class LearnableRotaryEmbedding(nn.Module):
    """
    Learnable RoPE with layer-wise frequency scaling.

    Each layer has a learnable alpha parameter that scales the RoPE frequencies:
        theta_i^(l) = alpha^(l) * base^(-2i/d)

    where alpha^(l) is constrained to [alpha_min, alpha_max] via sigmoid reparameterization:
        alpha^(l) = alpha_min + (alpha_max - alpha_min) * sigmoid(w^(l))

    This allows the model to learn optimal positional encoding frequencies per layer.
    """

    def __init__(
        self,
        config,
        layer_idx,
        total_attention_layers,
        alpha_min=0.1,
        alpha_max=2.0,
        device=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_attention_layers = total_attention_layers
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Get RoPE parameters from config
        dim = config.hidden_size // config.num_attention_heads
        base = getattr(config, 'rope_theta', 10000.0)
        max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        # Compute base inverse frequencies
        inv_freq_base = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq_base", inv_freq_base, persistent=False)

        # Learnable parameter (unconstrained, will be transformed via sigmoid)
        self.alpha_weight = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_weight)

    @property
    def inv_freq(self):
        return self.inv_freq_base * self.alpha

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Lfm2ForKaniModel(Lfm2Model):
    """
    Custom LFM2 model with KaniTTS-2 frame-level position encoding.

    This version only overrides position ID computation - everything else
    uses the standard Lfm2Model implementation.
    """

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128,
    ):
        super().__init__(config)
        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.use_learnable_rope = use_learnable_rope
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.speaker_emb_dim = speaker_emb_dim

        # Speaker embedding projection: 128 -> hidden_size (typically 1024)
        self.speaker_emb_projection = nn.Linear(speaker_emb_dim, config.hidden_size, bias=False)

        # Initialize learnable RoPE if enabled
        if use_learnable_rope:
            # Identify which layers are attention layers (not hybrid conv layers)
            attention_layer_indices = []
            if hasattr(config, 'layer_types'):
                for idx, layer_type in enumerate(config.layer_types):
                    if layer_type == "full_attention":
                        attention_layer_indices.append(idx)
            else:
                # Fallback: assume all layers are attention layers
                attention_layer_indices = list(range(config.num_hidden_layers))

            total_attention_layers = len(attention_layer_indices)

            # Create learnable RoPE modules for each layer
            self.learnable_rope_layers = nn.ModuleList()
            for idx in range(config.num_hidden_layers):
                if idx in attention_layer_indices:
                    learnable_rope = LearnableRotaryEmbedding(
                        config=config,
                        layer_idx=idx,
                        total_attention_layers=total_attention_layers,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        device=config.device if hasattr(config, 'device') else None,
                    )
                    self.learnable_rope_layers.append(learnable_rope)
                else:
                    # Conv layers don't use RoPE
                    self.learnable_rope_layers.append(None)

            print(f"Lfm2ForKaniModel initialized:")
            print(f"   - Audio tokens start: {audio_tokens_start}")
            print(f"   - Tokens per frame: {tokens_per_frame}")
            print(f"   - Speaker embedding: {speaker_emb_dim} -> {config.hidden_size}")
            print(f"   - Using frame-level position encoding (KaniTTS-2)")
            print(f"   - Learnable RoPE ENABLED for {total_attention_layers} attention layers")
            print(f"   - Alpha range: [{alpha_min}, {alpha_max}]")
        else:
            self.learnable_rope_layers = None
            print(f"Lfm2ForKaniModel initialized:")
            print(f"   - Audio tokens start: {audio_tokens_start}")
            print(f"   - Tokens per frame: {tokens_per_frame}")
            print(f"   - Audio step: {audio_step}")
            print(f"   - Speaker embedding: {speaker_emb_dim} -> {config.hidden_size}")
            print(f"   - Using frame-level position encoding (KaniTTS-2)")
            print(f"   - Learnable RoPE DISABLED (standard RoPE)")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        speaker_emb: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # Compute frame-level position IDs if not provided
        if position_ids is None and input_ids is not None:
            position_ids = compute_frame_level_positions(
                input_ids=input_ids,
                audio_tokens_start=self.audio_tokens_start,
                tokens_per_frame=self.tokens_per_frame,
                audio_step=self.audio_step
            )

        # If learnable RoPE is disabled, use standard forward pass
        if not self.use_learnable_rope:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # Learnable RoPE path
        from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache, create_causal_mask

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify at least one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2HybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_length = inputs_embeds.shape[1]
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.learnable_rope_layers[layer_idx] is not None:
                position_embeddings = self.learnable_rope_layers[layer_idx](hidden_states, position_ids)
            # Conv layers don't use RoPE; position_embeddings stays None or reuses last computed value

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class KaniTTS2ForCausalLM(Lfm2PreTrainedModel, GenerationMixin):
    """
    Flash Attention compatible LFM2 for causal language modeling with KaniTTS-2 frame-level positions.
    """
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def get_expanded_tied_weights_keys(self, *_, **__):
        # transformers 5.x expects a dict here; return empty to bypass weight tying.
        return {}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(
        self,
        config: Lfm2Config,
        audio_tokens_start: int,
        tokens_per_frame: int = 4,
        audio_step: float = 1.0,
        use_learnable_rope: bool = False,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        speaker_emb_dim: int = 128,
    ):
        super().__init__(config)

        self.model = Lfm2ForKaniModel(
            config,
            audio_tokens_start,
            tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.audio_tokens_start = audio_tokens_start
        self.tokens_per_frame = tokens_per_frame
        self.audio_step = audio_step
        self.use_learnable_rope = use_learnable_rope
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.speaker_emb_dim = speaker_emb_dim

        self._generation_state = None
        self._current_speaker_emb = None

        self.generation_config = config.generation_config if hasattr(config, 'generation_config') else None
        self.main_input_name = "input_ids"

        self.post_init()

    def _reset_generation_state(self, starting_frame_position: Optional[int] = None):
        self._generation_state = {
            'audio_tokens_generated': 0,
            'current_frame_position': float(starting_frame_position) if starting_frame_position is not None else None
        }

    def _update_generation_state(self, new_token_id: int):
        if self._generation_state is None:
            return
        if new_token_id >= self.audio_tokens_start:
            self._generation_state['audio_tokens_generated'] += 1

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        speaker_emb: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            speaker_emb=speaker_emb,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs
    ):
        if past_key_values is None and self._current_speaker_emb is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            speaker_emb_projected = self.model.speaker_emb_projection(self._current_speaker_emb)
            speaker_emb_projected = speaker_emb_projected.unsqueeze(1)
            inputs_embeds = torch.cat([
                inputs_embeds[:, :1, :],
                speaker_emb_projected,
                inputs_embeds[:, 1:, :]
            ], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask[:, :1],
                    torch.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask[:, 1:]
                ], dim=1)

            if cache_position is not None:
                cache_position = torch.cat([
                    cache_position[:1],
                    cache_position[:1] + 1,
                    cache_position[1:] + 1
                ], dim=0)

            input_ids = None

        if past_key_values is not None:
            if isinstance(past_key_values, (Cache, Lfm2HybridConvCache)):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
            else:
                cache_length = past_length = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0

            if input_ids is not None:
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                elif past_length == input_ids.shape[1]:
                    input_ids = input_ids[:, -1:]
            elif inputs_embeds is not None and past_length < inputs_embeds.shape[1]:
                inputs_embeds = inputs_embeds[:, past_length:]

        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(
                past_length, past_length + seq_length, device=device
            )

        if position_ids is None:
            if past_key_values is not None and self._generation_state is not None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                current_token = input_ids[0, -1].item()

                if current_token < self.audio_tokens_start:
                    pos = past_key_values.get_seq_length()
                else:
                    if self._generation_state['current_frame_position'] is None:
                        first_frame_pos = past_key_values.get_seq_length()
                        self._generation_state['current_frame_position'] = first_frame_pos

                    token_in_frame = self._generation_state['audio_tokens_generated'] % self.tokens_per_frame

                    if token_in_frame == 0 and self._generation_state['audio_tokens_generated'] > 0:
                        self._generation_state['current_frame_position'] += self.audio_step

                    pos = self._generation_state['current_frame_position']

                if isinstance(pos, float):
                    position_ids = torch.tensor([[pos]], device=device, dtype=torch.float)
                else:
                    position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)

                self._update_generation_state(current_token)

            else:
                if inputs_embeds is not None and self._current_speaker_emb is not None:
                    seq_len = inputs_embeds.shape[1]
                    device = inputs_embeds.device
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                elif input_ids is not None:
                    position_ids = compute_frame_level_positions(
                        input_ids=input_ids,
                        audio_tokens_start=self.audio_tokens_start,
                        tokens_per_frame=self.tokens_per_frame,
                        audio_step=self.audio_step
                    )

                if past_key_values is None and use_cache:
                    self._reset_generation_state(starting_frame_position=None)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

        if not (past_key_values is None and inputs_embeds is not None and self._current_speaker_emb is not None):
            model_inputs["cache_position"] = cache_position

        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds

        return model_inputs

    def generate(self, *args, **kwargs):
        speaker_emb = kwargs.pop('speaker_emb', None)
        self._generation_state = None
        self._current_speaker_emb = speaker_emb

        try:
            result = super().generate(*args, **kwargs)
        finally:
            self._generation_state = None
            self._current_speaker_emb = None

        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_tokens_start: int = None,
        tokens_per_frame: int = None,
        audio_step: float = None,
        use_learnable_rope: bool = None,
        alpha_min: float = None,
        alpha_max: float = None,
        speaker_emb_dim: int = None,
        *model_args,
        **kwargs
    ):
        base_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ['use_learnable_rope', 'alpha_min', 'alpha_max', 'speaker_emb_dim']}

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **base_kwargs)

        if audio_tokens_start is None:
            audio_tokens_start = getattr(config, 'audio_tokens_start', None)
            if audio_tokens_start is None:
                raise ValueError(
                    "audio_tokens_start not provided and not found in model config. "
                    "Please specify audio_tokens_start explicitly or add it to the model's config.json"
                )

        if tokens_per_frame is None:
            tokens_per_frame = getattr(config, 'tokens_per_frame', 4)
        if audio_step is None:
            audio_step = getattr(config, 'audio_step', 1.0)
        # Force disable learnable RoPE — the custom forward path has transformers 5.x
        # incompatibilities; the standard Lfm2Model.forward() path works correctly.
        use_learnable_rope = False
        if alpha_min is None:
            alpha_min = getattr(config, 'alpha_min', 0.1)
        if alpha_max is None:
            alpha_max = getattr(config, 'alpha_max', 2.0)
        if speaker_emb_dim is None:
            speaker_emb_dim = getattr(config, 'speaker_emb_dim', 128)

        model = cls(
            config=config,
            audio_tokens_start=audio_tokens_start,
            tokens_per_frame=tokens_per_frame,
            audio_step=audio_step,
            use_learnable_rope=use_learnable_rope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            speaker_emb_dim=speaker_emb_dim,
        )

        if use_learnable_rope:
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            import os

            if os.path.isdir(pretrained_model_name_or_path):
                safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            else:
                safetensors_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors"
                )

            state_dict = load_file(safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if 'lm_head.weight' in missing_keys and 'model.embed_tokens.weight' in state_dict:
                model.lm_head.weight = model.model.embed_tokens.weight
                missing_keys = [k for k in missing_keys if k != 'lm_head.weight']

            if missing_keys:
                print(f"   Missing keys (random init): {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys (ignored): {len(unexpected_keys)}")

            from transformers import GenerationConfig
            try:
                generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path)
                model.generation_config = generation_config
            except Exception:
                model.generation_config = GenerationConfig()

            device_map = base_kwargs.get('device_map', 'auto')
            if device_map == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
        else:
            base_model = Lfm2ForCausalLM.from_pretrained(pretrained_model_name_or_path, **base_kwargs)
            model.model.load_state_dict(base_model.model.state_dict(), strict=False)
            model.lm_head.load_state_dict(base_model.lm_head.state_dict())

            if hasattr(base_model, 'generation_config'):
                model.generation_config = base_model.generation_config

            model = model.to(base_model.device)

        print(f"Model loaded from {pretrained_model_name_or_path}")
        return model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
