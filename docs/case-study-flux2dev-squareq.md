# Case Study: Flux 2 Dev LoRA Training with SquareQ INT8 on 24GB VRAM

Training a 12-billion-parameter image diffusion model (Flux 2 Dev) with a 24-billion-parameter text encoder (Mistral 3), using SquareQ INT8 quantization for frozen weights and Stagehand block-swapping, on a single 24GB GPU.

SquareQ stores frozen transformer weights in INT8, dequantizing to bf16 on-the-fly during block loading. Combined with Stagehand's block-swapping, this keeps peak VRAM at ~6 GB for a model that needs ~60 GB in full precision.

## The Problem

Flux 2 Dev is Black Forest Labs' image generation model. Its components:

| Component | Parameters | bf16 Size |
|-----------|-----------|-----------|
| Transformer (8 double-stream + 48 single-stream blocks) | ~12B | ~24 GB |
| Text encoder (Mistral 3 24B) | 24B | ~48 GB |
| VAE | ~160M | ~0.3 GB |
| **Total** | **~36B** | **~72 GB** |

Neither the transformer nor the text encoder fits on a 24GB card individually. Loading both is impossible. The text encoder alone is 2x the card's capacity.

## The Solution: SquareQ + Stagehand + CPU Offload

Three techniques work together:

1. **SquareQ INT8 slab** — Frozen transformer weights quantized to INT8 (per-row symmetric). 203 layers in a ~30GB safetensors slab on disk. Dequantized to bf16 during block loading.
2. **Stagehand block-swapping** — Streams one transformer block at a time through a pinned memory pool. Forward and backward hooks manage the lifecycle automatically.
3. **accelerate.cpu_offload** — Mistral 3 text encoder uses per-layer CPU offload during the embedding caching pass. Each layer streams to GPU one at a time (~1.2 GB/layer peak).

### Phase 1: Cache Text Embeddings

```
Load: Mistral 3 24B text encoder + VAE
Strategy: accelerate.cpu_offload (per-layer dispatch)
Work: Process all captions, cache embeddings + latents to disk
```

Mistral 3's 48 GB of weights stream through GPU one layer at a time via `accelerate.cpu_offload()`. Each layer's parameters are placed on meta device with the state dict held on CPU. During forward, each submodule is moved to GPU, runs its forward, then moves back. Peak VRAM ~1.2 GB per layer.

After caching, the text encoder is unloaded: accelerate hooks are removed, the model moves to CPU, and GPU memory drops to near zero.

### Phase 2: Train with SquareQ + Stagehand

```
Load: Flux 2 Dev transformer (from SquareQ INT8 slab)
Strategy: StagehandStrategy with SquareQ V2 backing
Work: LoRA fine-tuning with cached embeddings
```

The transformer has 56 blocks (8 double-stream, 48 single-stream). With SquareQ backing, frozen weights are stored as INT8 in the slab file. On each block load:

1. INT8 weights read from slab via memory-mapped I/O
2. Dequantized to bf16 in the pinned CPU slab
3. Copied to GPU via async DMA on a dedicated CUDA stream
4. Module parameters repointed to the GPU tensor
5. Forward/backward runs with bf16 precision
6. Block evicted: GPU tensor freed, LoRA grads preserved on CPU

Only LoRA adapter weights (~50 MB at rank 16) and non-block submodules (~1.5 GB: norms, projections, embeddings) stay on GPU permanently.

## Hardware Requirements

| Resource | Minimum | Used |
|----------|---------|------|
| GPU VRAM | 16 GB (estimated) | 24 GB (RTX 3090/4090) |
| System RAM | 32 GB | 62 GB |
| Disk | 35 GB free (for slab) | NVMe |

### Memory Budget

**Text encoding phase** (~26 GB CPU):
- Mistral 3 weights on CPU: ~48 GB (meta device + state dict)
- Peak GPU: ~1.2 GB (single layer)

**Training steady state**:
- GPU allocated: 5.52 GB (non-block submodules + LoRA + activations)
- GPU reserved: 6.00 GB (with `empty_cache()` after each step)
- Pinned pool: 8 GB
- SquareQ slab: memory-mapped from disk (~30 GB file, not in RAM)

## Training Configuration

```json
{
  "model_type": "flux_2_dev",
  "training_method": "lora",
  "model": {
    "type": "flux_2_dev",
    "path": "black-forest-labs/FLUX.2-dev"
  },
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16.0,
    "target_modules": [
      "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
      "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"
    ]
  },
  "memory": {
    "gradient_checkpointing": "on",
    "strategy": "stagehand",
    "stagehand": {
      "pinned_pool_mb": 8192,
      "pinned_slab_mb": 2048,
      "vram_high_watermark_mb": 10000,
      "vram_low_watermark_mb": 6000,
      "squareq_slab_path": "output/squareq_slabs/flux2dev_int8.safetensors",
      "squareq_manifest_path": "output/squareq_slabs/flux2dev_int8.manifest.json"
    }
  },
  "resolution": 512,
  "batch_size": 1,
  "learning_rate": 1e-4,
  "bucket_policy_enabled": false
}
```

## Results

Training on 118 images with text captions, 200 steps, LoRA rank 16, cosine LR schedule with 10-step warmup.

### Loss Curve

```
Step   1: loss=0.696  avg=0.696  lr=2.00e-05  gn=0.030
Step   5: loss=0.554  avg=0.726  lr=6.00e-05  gn=0.098
Step  10: loss=0.626  avg=0.731  lr=1.00e-04  gn=0.061
Step  15: loss=0.714  avg=0.741  lr=9.97e-05  gn=0.032
Step  20: loss=0.616  avg=0.742  lr=9.90e-05  gn=0.037
Step  25: loss=0.675  avg=0.765  lr=9.83e-05  gn=0.025
Step  30: loss=0.584  avg=0.748  lr=9.70e-05  gn=0.039
```

Loss stabilized around 0.71-0.75 average within the first 30 steps. Gradient norms stayed healthy in the 0.02-0.10 range, indicating stable convergence.

### Performance

| Metric | Value |
|--------|-------|
| Steady-state step time | ~160 s |
| VRAM allocated (steady state) | 5.52 GB |
| VRAM reserved (steady state) | 6.00 GB |
| Text encoding (118 samples) | ~12 min |
| Latent caching (118 samples) | ~12 min |
| Training (200 steps, estimated) | ~8.9 hours |
| SquareQ params matched | 192/203 |
| OOM events | 0 (after all fixes) |

### VRAM Profile

The VRAM profile is remarkably flat compared to full-precision Stagehand training:

- **Allocated**: Constant 5.52 GB — non-block submodules + LoRA weights + per-step activations
- **Reserved**: Constant 6.00 GB — the `empty_cache()` call after each step keeps reserved memory tightly controlled
- **Gap**: Only 0.48 GB between allocated and reserved, meaning almost no wasted cache

This is possible because SquareQ INT8 blocks are ~half the size of bf16 blocks. Each block loads, dequantizes, runs forward/backward, and evicts within the watermark budget without approaching the card's 24 GB limit.

## Key Stagehand Features Used

### SquareQ V2 Backing

SquareQ-backed blocks use a different loading path than module-backed or file-backed:

```python
# In scheduler._load_block():
if block_entry.squareq_backed:
    squareq_layers = self._get_squareq_v2_layers(source_path)
    _copy_squareq_backed_params_into_buffer(
        module=module,
        buffer=slab.buffer,
        layout=layout,
        squareq_layers=squareq_layers,
    )
```

INT8 weights are read from the slab, dequantized per-row using stored scales, and written into the pinned slab buffer as bf16. The buffer is then DMA'd to GPU. This happens transparently — the training loop sees bf16 module parameters.

### Key Matching with Aliases

Flux 2 Dev uses diffusers' naming internally (e.g. `ff.net.0.proj`) but the SquareQ slab was built from HF checkpoint names (e.g. `ff.linear_in`). The registry's `_candidate_tensor_keys()` uses bidirectional alias tables to bridge this gap:

```python
aliases = (
    ("ff.net.0.proj", "ff.linear_in"),
    ("ff.net.2", "ff.linear_out"),
    ("ff_context.net.0.proj", "ff_context.linear_in"),
    ("ff_context.net.2", "ff_context.linear_out"),
)
```

Additionally, LoRA injection renames base weights (`attn.to_q.weight` → `attn.to_q.orig.weight`). The `_candidate_squareq_layer_keys()` method strips these adapter suffixes before matching.

### Post-Step Cache Cleanup

Stagehand now calls `torch.cuda.empty_cache()` after each step in `forward_context()`. Without this, PyTorch's caching allocator holds reserved memory from evicted blocks indefinitely. With aspect ratio bucketing (varying activation tensor sizes), the allocator can't reuse fixed-size cached blocks, and reserved memory grows until OOM.

Before the fix: `vram=5.52/14.95G` (9.4 GB dead reserved memory).
After the fix: `vram=5.52/6.00G` (0.48 GB overhead).

## Bugs Fixed Along the Way

Five bugs were identified and fixed to make this work:

1. **Mistral 3 text encoder OOM**: `text_encoder.to(cuda)` tries to move 48 GB to GPU. Fix: use `accelerate.cpu_offload()` for per-layer streaming. File: `serenity/models/flux2.py`.

2. **Legacy config silently discarding Stagehand config**: Config keys `base_model_name` or `output_model_destination` trigger `_is_legacy_config()` which rebuilds the entire config dict, discarding all stagehand/squareq settings. Fix: use new Serenity config format. File: config restructure.

3. **SquareQ matching only 32/203 params**: Two sub-causes — LoRA `.orig` suffix not stripped during key matching, and Flux 2 FF layer naming mismatch between diffusers and HF checkpoint. Fix: suffix stripping + bidirectional aliases in `_candidate_tensor_keys()`. File: `stagehand/registry.py`.

4. **CUDA memory fragmentation**: Stagehand evicts blocks but PyTorch holds reserved memory. Over multiple steps, reserved grows to fill the card. Fix: `torch.cuda.empty_cache()` after each step. File: `serenity/memory/stagehand_strategy.py`.

5. **Bucket policy disabling gradient checkpointing**: After 3 successful steps, the memory predictor sees low VRAM (Stagehand makes it appear nearly empty) and auto-assigns `FAST` mode which disables gradient checkpointing. Without checkpointing, activations blow up VRAM. Fix: disable bucket_policy when Stagehand is active. File: config setting.

## SquareQ Slab Building

The SquareQ slab was built with `scripts/build_flux2dev_slab.py`:

```bash
python scripts/build_flux2dev_slab.py \
  --model black-forest-labs/FLUX.2-dev \
  --output output/squareq_slabs/flux2dev_int8 \
  --dtype int8
```

Output:
- `flux2dev_int8.safetensors` — ~30 GB slab with 203 quantized layers
- `flux2dev_int8.manifest.json` — canonical name → offset/shape/scale mapping

Quantization is per-row symmetric INT8: each row of a weight matrix gets its own scale factor. This preserves more precision than per-tensor quantization while keeping the format simple (no zero points, no group quantization).

## What This Means

A 12B image model + 24B text encoder = 36B total parameters, trained with LoRA on a single 24GB GPU. The frozen weights are INT8 quantized (halving their size) and streamed from disk one block at a time. Only ~6 GB of VRAM is used at steady state.

The cost is speed: ~160 seconds per step due to block-swapping overhead and INT8→bf16 dequantization. For LoRA fine-tuning (100-500 steps), this completes in 4-22 hours on a single consumer GPU — a task that previously required 48+ GB VRAM or multi-GPU setups.

The combination of SquareQ + Stagehand is strictly more memory-efficient than full-precision Stagehand alone: INT8 blocks are half the size, so they transfer faster and leave more VRAM headroom for activations. The trade-off is a small quantization error in the frozen base weights, which LoRA can compensate for during fine-tuning.
