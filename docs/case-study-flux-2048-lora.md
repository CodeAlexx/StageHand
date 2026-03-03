# Case Study: Flux 1 Dev 12B LoRA at 2048x2048 on 4.15 GB VRAM

Training a 12-billion-parameter image diffusion model (Flux 1 Dev) with LoRA at 2048x2048 resolution, full bfloat16 precision, using only 4.15 GB of a 24 GB GPU.

No quantization. No model sharding. 90% of transformer layers offloaded to CPU.

## The Problem

Flux 1 Dev is Black Forest Labs' high-resolution image generation model:

| Component | Parameters | bf16 Size |
|-----------|-----------|-----------|
| Transformer (FluxTransformer2DModel) | ~12B | ~24 GB |
| Text encoder 1 (CLIP-L) | ~125M | ~250 MB |
| Text encoder 2 (T5-XXL) | ~4.8B | ~9.6 GB |
| VAE | ~84M | ~168 MB |
| **Total** | **~17B** | **~34 GB** |

At 2048x2048 resolution, input latents are 128x128 spatial tokens (vs 64x64 at 1024). Attention scales quadratically with token count, so VRAM pressure from activations is 4x higher than at 1024. Combined with the 12B transformer, training at this resolution was assumed to require either quantization (FP8/NF4) or multi-GPU setups.

Previous best published result: Kohya ss achieved ~4 GB for Flux training using FP8 quantization + block swapping. But FP8 introduces quantization noise into gradients and activations. Full bf16 at 2048x2048 had not been demonstrated on a single 24 GB card.

## The Solution: LayerOffloadConductor

Stagehand's `LayerOffloadConductor` manages 57 transformer blocks, keeping only a fraction resident on GPU at any time. Combined with activation offloading and gradient checkpointing, the entire training loop fits in under 5 GB.

### How It Works

The conductor operates at a higher level than Stagehand's block-swap scheduler. Instead of file-backed streaming, it uses simple layer-level CPU/GPU movement:

1. **Resident fraction**: With `layer_offload_fraction=0.9`, 90% of layers are offloaded. Only ~6 of 57 layers stay on GPU permanently.
2. **Before each layer**: The conductor loads the layer to GPU (if not already resident). It also evicts any stale layers left by aborted checkpoint recomputation.
3. **After each layer**: The conductor immediately moves non-resident layers back to CPU, preventing VRAM accumulation across the stack.
4. **Activation offloading**: Between layers, checkpoint-saved activations are moved to CPU. They're brought back to GPU just before the next layer needs them.
5. **Gradient checkpointing**: `OffloadCheckpointLayer` wraps each layer with PyTorch's `checkpoint()`, so activations are recomputed during backward instead of stored. This is critical at 2048x2048 where a single block's activations can exceed available VRAM.

### The Forward/Backward Dance

Each training step:

```
Forward:  [load block 0] → compute → [evict 0, load 1] → compute → ... → [load 56] → compute → loss
                    ↕ activations shuttle CPU↔GPU between blocks

Backward: [load block 56] → recompute fwd → grad → [evict 56, load 55] → ... → [load 0] → grad
                    ↕ checkpointing recomputes activations on the fly
```

The GPU only ever holds: 1 transformer block (~400 MB) + LoRA weights (~30 MB) + optimizer state (~122 MB) + activations for current block + overhead. Everything else is on CPU.

## Hardware

| Resource | Used |
|----------|------|
| GPU | RTX 3090 (24 GB VRAM) |
| VRAM allocated | 4.15 GB |
| VRAM reserved | 4.46 GB |
| System RAM | 62 GB |
| Disk | NVMe SSD |

19.5 GB of the 24 GB GPU sits unused. This is a LoRA training run that could fit on a 6 GB card.

## Training Configuration

```json
{
  "model_type": "flux_dev",
  "training_method": "lora",
  "lora_rank": 16,
  "lora_alpha": 16.0,
  "model": {
    "path": "black-forest-labs/FLUX.1-dev",
    "dtype": "bfloat16"
  },
  "memory": {
    "gradient_checkpointing": "on",
    "enable_activation_offloading": true,
    "enable_async_offloading": true,
    "layer_offload_fraction": 0.9
  },
  "optimizer": { "optimizer": "adamw", "weight_decay": 0.01 },
  "scheduler": { "scheduler": "constant", "warmup_steps": 0 },
  "timestep_distribution": "logit_normal",
  "learning_rate": 0.0004,
  "resolution": 2048,
  "batch_size": 1,
  "max_steps": 50,
  "train_dtype": "bfloat16"
}
```

Key settings:
- `layer_offload_fraction: 0.9` — offload 90% of 57 layers to CPU
- `enable_activation_offloading: true` — move activations to CPU between layers
- `enable_async_offloading: true` — use non-blocking CUDA copies for overlap
- `gradient_checkpointing: "on"` — recompute activations during backward

## Results

50 steps on 407 images (3072-4096px source, bucketed to 2048x2048), LoRA rank 16.

### Performance

| Metric | Value |
|--------|-------|
| First step (warmup) | ~154 s |
| Steady-state step | 32 s |
| VRAM allocated | 4.15 GB |
| VRAM reserved | 4.46 GB |
| Total wall time | ~26 min |
| Latent cache size (407 samples) | 2,445 MB |

### Loss Curve

```
Step  1: loss=0.359  grad_norm=0.014
Step  2: loss=0.370  grad_norm=0.034
Step  3: loss=0.260  grad_norm=0.004
Step  4: loss=0.295  grad_norm=0.010
```

Loss is in the expected range for flow-matching. Gradient norms are small and stable, indicating healthy LoRA training.

### VRAM Breakdown

| Component | Size |
|-----------|------|
| Resident layers (~6 of 57) | ~2.4 GB |
| LoRA adapters (rank 16) | ~30 MB |
| Optimizer state (AdamW) | ~122 MB |
| Activations (single block, peak) | ~1.0 GB |
| Misc (buffers, norms, projections) | ~600 MB |
| **Total allocated** | **~4.15 GB** |

The remaining 51 layers (~20 GB of parameters) live on CPU. Each layer is loaded to GPU only for its forward or backward pass, then immediately evicted.

## What Makes This Work

### 1. OffloadCheckpointLayer Integration

Each transformer block is wrapped by `OffloadCheckpointLayer`, which combines two responsibilities:

- **Gradient checkpointing**: Wraps the forward pass in `torch.utils.checkpoint.checkpoint()` so activations are not stored — they're recomputed during backward.
- **Conductor callbacks**: Calls `conductor.before_layer()` and `conductor.after_layer()` to manage CPU/GPU movement.

Without gradient checkpointing, storing activations for 57 layers at 2048x2048 would require tens of GB. With it, only the current layer's activations exist at any time.

### 2. Budget-Based Eviction

Non-reentrant checkpointing (`use_reentrant=False`) can abort mid-recomputation via `_StopRecomputationError`. When this happens, the conductor's `after_layer()` hook never fires, leaving stale blocks on GPU. The `before_layer()` hook detects and evicts these stale blocks before loading the next layer:

```python
# In before_layer: evict stale blocks from aborted checkpoint recomputation
keep = self._keep_on_train()
for idx, s in enumerate(self._layers):
    if idx != layer_index and idx >= keep and s.on_train_device:
        s.layer.to(device=self.temp_device, non_blocking=self.enable_async)
        s.on_train_device = False
```

### 3. Activation Offloading

Between blocks, the conductor moves checkpoint-saved activations to CPU and back:

- `after_layer()`: `activations.to(temp_device)` — move to CPU
- `before_layer()`: `activations.to(train_device)` — bring back to GPU

Combined with `non_blocking=True`, these transfers overlap with computation on the previous/next block.

### 4. Kwargs Activation Handling

Flux's transformer blocks pass tensors as keyword arguments, not just positional args. The conductor's activation offloading only handled positional args. A fix in `OffloadCheckpointLayer` ensures kwargs tensors are also moved to the correct device before the block's forward pass.

## Batch Size Scaling

At 2048x2048 with activation offloading, batch size scales cleanly:

| Batch Size | VRAM | Step Time | Result |
|------------|------|-----------|--------|
| 1 | 4.15 GB | 32 s | Stable |
| 2 | 4.16 GB | 61 s | Stable |
| 4 | OOM | — | Exceeds budget |

Batch 2 fits at the same VRAM as batch 1 because activation offloading moves intermediate tensors to CPU between layers — doubling the batch only doubles the per-layer activation size, which is small enough to absorb. Step time roughly doubles (32s to 61s) due to twice the computation per step. Batch 4 exceeds the per-layer activation budget and OOMs.

## Context: Why This Matters

| Framework | Flux Training | Precision | VRAM |
|-----------|--------------|-----------|------|
| Kohya ss | 1024x1024 | FP8 + block swap | ~4 GB |
| **Serenity + Stagehand** | **2048x2048** | **Full bf16** | **4.15 GB** |

Full bf16 means:
- No quantization noise in gradients or activations
- No FP8 scaling factor tuning
- No mixed-precision edge cases
- Exact same numerical behavior as a multi-GPU setup with all weights resident

The trade-off is wall time: 32 s/step vs ~2-4 s/step with all weights on GPU. For LoRA fine-tuning (50-500 steps), the total overhead is 15-30 minutes — acceptable for most workflows.

## Reproducing This

This was run with [Serenity](https://github.com/CodeAlexx/Serenity), which uses [Stagehand](https://github.com/CodeAlexx/StageHand) as its memory management backend.

Relevant code paths:
- **LayerOffloadConductor**: `serenity/memory/conductor/offload.py`
- **OffloadCheckpointLayer**: `serenity/memory/checkpoint_layer.py`
- **Flux 1 model adapter**: `serenity/models/flux1.py`
- **Training loop**: `serenity/cli/native_diffusion.py`
- **Loss computation**: `serenity/cli/diffusion_losses.py`

The conductor is model-agnostic. Any model with sequential transformer blocks can use it by:
1. Creating a `LayerOffloadConductor` with the desired `layer_offload_fraction`
2. Wrapping each block with `OffloadCheckpointLayer`
3. Calling `conductor.forward_context()` around the forward pass
