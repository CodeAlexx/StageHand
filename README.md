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
