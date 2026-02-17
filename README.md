# Stagehand

**GPU memory orchestrator for large model inference and training.**

Run 64GB+ models on 24GB GPUs. No quantization. No quality loss.

## What It Does

Stagehand manages GPU VRAM like a stage manager manages actors: loading what's needed, unloading what's not, and prefetching what's next.

- **Block-swap streaming**: Stream transformer blocks through VRAM one at a time via file-backed sources
- **Pinned memory pools**: Pre-allocated CPU↔GPU transfer lanes for zero-allocation copies
- **Async prefetch**: Overlap compute and data transfer so the GPU stalls less
- **VRAM budget tracking**: Hard budget enforcement with automatic eviction
- **File-backed registry**: Weights can stay on disk instead of full CPU RAM residency

## Install

```bash
pip install stagehand
```

## Quick Start

```python
import torch
from stagehand import StagehandConfig, StagehandRuntime

model = ...  # torch.nn.Module

cfg = StagehandConfig(
    pinned_pool_mb=8192,
    pinned_slab_mb=2048,
    vram_high_watermark_mb=20000,
    vram_low_watermark_mb=16000,
    prefetch_window_blocks=2,
    max_inflight_transfers=1,
)

runtime = StagehandRuntime(
    model=model,
    config=cfg,
    block_pattern=r"^(transformer_blocks\.\d+|single_transformer_blocks\.\d+)$",
    group="transformer",
    dtype=torch.bfloat16,
    inference_mode=True,
)

runtime.convert_registry_to_file_backed_sharded("/path/to/transformer/shards")

runtime.begin_step(0)
runtime.before_block("transformer_blocks.0")
# run the block...
runtime.after_block("transformer_blocks.0", output_tensor)
runtime.end_step()
```

## Layer Mode (NEW)

One line, zero config, works on **any** model:

```python
import stagehand

model = YourModel()
model = stagehand.layer(model)

# Just use the model normally — Stagehand handles the rest
for batch in dataloader:
    output = model(batch)
```

Layer mode wraps individual `nn.Linear`, `nn.Conv2d`, and `nn.Embedding` modules and manages CPU/GPU transfer through a bounded pinned pool. It uses a two-phase lifecycle:

1. **Trace** (step 0): Records actual execution order with zero prefetch
2. **Scheduled** (step 1+): Rebuilds the registry with traced order and enables k-lookahead prefetch

Auto-step detection means you never call `begin_step()`/`end_step()` — Stagehand detects when a new forward pass begins.

### Configuration

```python
# Explicit VRAM budget
model = stagehand.layer(model, vram_budget="8GB")

# Custom prefetch depth
model = stagehand.layer(model, prefetch_k=5)

# Inference mode (no backward hooks)
model = stagehand.layer(model, inference_mode=True)

# Reusable config
rt = stagehand.Runtime(vram_budget="4GB", prefetch_k=5)
model = rt.layer(model)
```

### Shutdown and Pool Reuse

```python
runtime = model._stagehand_layer_runtime

# Clean shutdown
runtime.shutdown()

# Or keep pool for reuse
pool = runtime.shutdown_keep_pool()
model2 = stagehand.layer(other_model, pool=pool)
```

## Block Mode

For transformer models with known block patterns, block mode gives you explicit control:

```python
from stagehand import StagehandConfig, StagehandRuntime

cfg = StagehandConfig(
    pinned_pool_mb=8192,
    pinned_slab_mb=2048,
    vram_high_watermark_mb=20000,
    vram_low_watermark_mb=16000,
    prefetch_window_blocks=2,
)

runtime = StagehandRuntime(
    model=model,
    config=cfg,
    block_pattern=r"^transformer_blocks\.\d+$",
    inference_mode=True,
)
```

## Inference Eviction Behavior

When `inference_mode=True`, Stagehand now uses eviction save-back policy by block type:

- **File-backed / SquareQ-backed blocks**: `save_back=False` (reload from source on demand)
- **Module-backed blocks**: `save_back=True` (prevents detached-empty parameter tensors after eviction/reload)

This means inference is safe in both modes:

- file-backed streaming (`convert_registry_to_file_backed_sharded(...)`) for lowest RAM pressure
- module-backed inference when you need in-memory parameters

## Comparison with mmgp

| Feature | Stagehand | mmgp |
|---------|-----------|------|
| File-backed mmap streaming | ✅ | ❌ |
| Async prefetch overlap | ✅ | Partial |
| VRAM budget enforcement | ✅ | ✅ |
| Pinned memory pools | ✅ | ✅ |
| CPU RAM requirement | Low (file-backed mode) | Higher |
| PyPI package | ✅ | ✅ |

## License

MIT
