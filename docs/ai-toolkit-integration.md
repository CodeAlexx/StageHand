# ai-toolkit Integration

Stagehand integrates with [ai-toolkit](https://github.com/ostris/ai-toolkit) to provide block-level CPU/GPU weight swapping for FLUX model training on limited VRAM.

## Architecture

The integration uses a dual-backend approach:

- **Training**: Kohya-style block swap (fast, ~4s/step on FLUX 12B) or `stagehand.layer()` fallback (~50s/step)
- **Sampling**: Turbo Rust engine when available, otherwise training hooks stay active

The `StagehandManager` in `toolkit/stagehand_conductor.py` orchestrates mode transitions between training and sampling via `enter_sampling_mode()` / `exit_sampling_mode()`.

### Kohya block swap

Replicates the block-swap approach from kohya-ss/sd-scripts:
- Block-level (not per-linear) weight swapping via secondary CUDA stream
- GPU tensor reuse: outgoing block's GPU memory becomes incoming block's
- Forward hooks (wait/submit) + backward hooks for gradient flow
- Non-block layers (embedders, norms, final projections) stay permanently on GPU

### Turbo sampling

During sampling, training hooks are suspended and the Rust Turbo engine takes over:
1. `enter_sampling_mode()` removes Kohya forward hooks
2. Turbo `before_block()` / `after_block()` hooks are installed
3. Non-block layers are moved to GPU
4. After sampling, `exit_sampling_mode()` restores Kohya hooks and block positions

## Configuration

```yaml
model:
  name_or_path: black-forest-labs/FLUX.1-dev
  is_flux: true
  stagehand:
    enabled: true
    vram_budget_gb: 18.0      # VRAM ceiling for Turbo pool sizing
    use_turbo: true            # Use Rust engine for sampling (falls back if unavailable)
    block_swap_mode: kohya     # "kohya" (fast) or "stagehand" (layer-mode fallback)
    blocks_to_swap: 30         # Number of blocks to keep on CPU during training
```

### VRAM profiles

| GPU | `vram_budget_gb` | `blocks_to_swap` | Notes |
|-----|------------------|-------------------|-------|
| 24 GB | 18.0 | 30 | Comfortable for FLUX 12B LoRA |
| 12 GB | 9.0 | 40 | Tight, may need smaller batch/resolution |

## Example configs

- `config/examples/train_lora_flux_stagehand_24gb.yaml` — 24 GB GPU, Kohya swap + Turbo
- `config/examples/train_lora_flux_stagehand_12gb.yaml` — 12 GB GPU, aggressive swap

## Install

```bash
pip install stagehand
# Optional: Turbo Rust engine for fast sampling
pip install stagehand[turbo]
```

## Files

| File | Purpose |
|------|---------|
| `toolkit/stagehand_conductor.py` | `StagehandManager` — unified training/sampling orchestration |
| `toolkit/kohya_block_swap.py` | `KohyaBlockSwap` — Kohya-style block-level CPU/GPU swap |
| `toolkit/config_modules.py` | `StagehandConfig` dataclass |
| `toolkit/stable_diffusion_model.py` | `apply_stagehand()` + sampling integration |
| `extensions_built_in/sd_trainer/SDTrainer.py` | `prepare_before_forward()` call in training loop |
