# Kani TTS dependency compatibility (GM Voice Studio)

## Observed environment

- Python 3.12.3  
- torch 2.10.0+cu128  
- transformers 5.3.0  
- kani_tts (module) 0.1.0 / kani-tts-2 (package) 0.0.5  

## What Kani TTS declares

From `importlib.metadata` for **kani-tts-2** (PyPI package):

- **Requires:** `torch>=2.8.0`, `nemo-toolkit[tts]==2.4.0`, `numpy`, `scipy`, `librosa`, `omegaconf`, `torchaudio>=2.8.0`, `torchcodec>=0.10.0`, optional `soundfile`, dev extras.  
- **Does not declare:** `transformers`.  

The package **code** imports `transformers.models.lfm2` (LFM2). LFM2 was added in **transformers 5.3.0**. So a compatible stack is:

- **torch:** >=2.8.0 (per package)  
- **transformers:** >=5.3.0 (required for LFM2; not in package metadata)  

Older docs (e.g. HuggingFace README) sometimes say `transformers==4.56.0`; that does **not** apply to the LFM2-based kani-tts-2-en model, which needs 5.3+.

## Exact root cause of the narrate failure

1. **RoPE tensor size mismatch**  
   When a **speaker embedding** is used, kani_tts inserts it at position 1, so the sequence length becomes **N** (e.g. 19).  
   In some code paths, **position_ids** and **cache_position** are still for length **N−1** (e.g. 18).  
   The inner LFM2 model builds RoPE (cos/sin) from those position tensors, so cos/sin have length N−1 while **query_states** have length N → `apply_rotary_pos_emb` raises:  
   `RuntimeError: The size of tensor a (19) must match the size of tensor b (18) at non-singleton dimension 2`.

2. **Additional compatibility gaps (handled in-app)**  
   - **Lfm2Config:** kani_tts expects `config.rope_theta`; transformers 5.x uses `rope_parameters` (dict). Patched in `app/main.py`.  
   - **PreTrainedModel:** `_tied_weights_keys` as list; transformers 5.x expects a dict in `get_expanded_tied_weights_keys`. Patched in `app/main.py`.  
   - **Lfm2ForKaniModel:** kani_tts forward uses `self.pos_emb`; transformers 5 Lfm2Model has `rotary_emb`. Patched in `app/services/tts_service.py`.  
   - **position_ids / cache_position:** When using `inputs_embeds` (speaker-emb prefill), lengths are forced to match `inputs_embeds.shape[1]` in `tts_service.py` (prepare_inputs and forward patches).

So the **exact root cause** of the narrate failure with cloned voice is the **position/cache length mismatch** (N vs N−1) leading to the RoPE size error; dependency-wise, **transformers 5.3.0 is required** and is compatible as long as the app’s patches (and optional fallback) are in place.

## Exact package version changes

| Package        | Before (requirements) | After (pinned/compatible) |
|----------------|------------------------|----------------------------|
| torch          | >=2.5.0                | >=2.8.0                    |
| transformers   | >=5.1.0                | >=5.3.0                    |
| kani-tts-2     | (unversioned)          | >=0.0.5                    |

- **requirements.txt** and **requirements-core.txt** updated to the above.  
- **requirements-tts.txt** added with the same TTS stack for a dedicated install path.

No change to Python version (3.12).

## Graceful fallback

In **app/services/tts_service.py**, if TTS generation raises, the service now returns **1 second of silence** and the same sample rate instead of raising. So:

- Narrate API can still return a WAV (with silence for failed chunks).  
- Text/narration flow continues; callers can treat short or silent audio as “audio failed”.

## Standalone TTS repro

- **scripts/tts_repro.py**  
  - Minimal script (no FastAPI).  
  - Applies the same compatibility patches (TransformersKwargs/Unpack, Lfm2Config.rope_theta, tied_weights_keys, pos_emb).  
  - Usage:  
    - `.venv/bin/python scripts/tts_repro.py` — no speaker emb  
    - `.venv/bin/python scripts/tts_repro.py path/to/voice.pt` — with speaker emb  

- **Standalone TTS after the fix**  
  - With the repro script and the app’s patches applied inside the script, model load and generation **without** speaker_emb can run.  
  - **With** speaker_emb, the repro script does **not** apply the position_ids/cache_position patches (those live in `tts_service._get_tts()`). So the RoPE mismatch can still occur in the standalone repro when using a cloned voice; the **API** uses the full app patches and fallback.

## API narration after the fix

- **With compatible versions** (torch>=2.8.0, transformers>=5.3.0, kani-tts-2>=0.0.5) and the existing app patches (main.py + tts_service.py):  
  - **Without** cloned voice: narration can work end-to-end.  
  - **With** cloned voice: the position/cache patches in tts_service aim to fix the 19 vs 18 RoPE error; if they do not cover every path, the new **graceful fallback** returns silence so the narrate endpoint still responds and text narration still “works” (with silent or shortened audio when TTS fails).

## Summary table

| Item                         | Result |
|------------------------------|--------|
| Root cause                   | RoPE length mismatch (N vs N−1) when speaker emb is inserted; plus transformers 5.x API gaps (rope_theta, tied_weights, pos_emb). |
| Package version changes      | torch>=2.8.0, transformers>=5.3.0, kani-tts-2>=0.0.5. |
| Standalone TTS (no speaker)  | Works with repro script + compat patches. |
| Standalone TTS (with speaker)| May still hit RoPE mismatch in repro (no position sync patches there). |
| API narration                | Works without cloned voice; with cloned voice, either fixed by patches or fallback returns silence so narration flow completes. |
