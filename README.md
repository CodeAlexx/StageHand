# Stagehand

GPU memory orchestrator for PyTorch. Streams model weights between CPU and GPU so models larger than VRAM can run without quantization.

**Status**: Alpha (`0.1.0`). Five commits. API may change. Used in [Serenity](https://github.com/CodeAlexx/Serenity) for diffusion model training.

## How it works

Stagehand keeps most model weights on CPU (in pinned memory or on disk) and streams them to GPU one layer/block at a time, through a fixed-size pool. The core loop:

1. **Pre-forward hook fires** → load this layer's weights from pinned CPU slab to GPU via async DMA copy on a dedicated CUDA stream
2. **Forward runs** on GPU with weights resident
3. **Post-forward hook fires** → mark layer done, decrement refcount
4. **Eviction** → when VRAM usage crosses a high watermark, evict least-soon-needed layers back to CPU (or drop them if they can be reloaded from disk)
5. **Prefetch** → after each layer, issue async copies for the next *k* layers so they arrive before they're needed

The GPU never runs out of memory because the pool is bounded and eviction is enforced.

### What actually happens to the weights

- **Module-backed** (default): Weights live in CPU RAM. On load: copy to GPU tensor, repoint `module.weight.data` into it. On eviction: copy back to CPU, repoint back. This is a full round-trip every time.
- **File-backed**: Weights live on disk as safetensors. On load: mmap read → pinned slab → GPU. On eviction (inference): just drop the GPU tensor — weights reload from disk on next use. No CPU RAM needed for frozen weights.
- **SquareQ-backed**: INT8 quantized format (`.fpk`/`.slab`). Dequantized to target dtype during host staging. Same eviction behavior as file-backed.

## Two modes

### Layer mode — zero config

```python
import stagehand

model = stagehand.layer(model)

# Use normally. Stagehand handles everything.
output = model(input)
```

Wraps every `nn.Linear`, `nn.Conv2d`, and `nn.Embedding` in the model. Everything else (LayerNorm, biases, buffers) stays on GPU permanently — they're small.

**Two-phase lifecycle:**
- **Step 0 (trace)**: No prefetch. Forward hooks record actual execution order. When the first layer fires a second time, trace completes automatically.
- **Step 1+ (scheduled)**: Registry rebuilt with traced order. `prefetch_k` layers are loaded ahead. Auto-step detection — you never call `begin_step()`/`end_step()`.

**Options:**
```python
stagehand.layer(model,
    vram_budget="8GB",      # VRAM cap (default: 80% of detected)
    ram_budget="16GB",      # Pinned pool cap
    prefetch_k=3,           # Lookahead depth (default: 3)
    dtype=torch.bfloat16,   # Storage dtype (default: bfloat16)
    inference_mode=True,    # Skip backward hooks
    telemetry=False,        # Disable JSONL logging
)
```

**Pool auto-sizing**: Slab size = next power-of-2 MiB above the largest layer. Pool = `(prefetch_k + 4)` slabs. VRAM watermarks: 80% high, 60% low of detected VRAM.

### Block mode — explicit control

For transformer models where you know the block structure:

```python
from stagehand import StagehandConfig, StagehandRuntime

cfg = StagehandConfig(
    pinned_pool_mb=8192,         # 8 GB pinned pool
    pinned_slab_mb=2048,         # 2 GB per slab (= 4 slabs)
    vram_high_watermark_mb=20000,
    vram_low_watermark_mb=16000,
    prefetch_window_blocks=2,
)

runtime = StagehandRuntime(
    model=model,
    config=cfg,
    block_pattern=r"^transformer_blocks\.\d+$",
    dtype=torch.bfloat16,
    inference_mode=True,
)

# File-backed: weights stream from disk, not CPU RAM
runtime.convert_registry_to_file_backed_sharded("/path/to/shards")

# Manual step lifecycle
with runtime.managed_forward():
    output = model(input)
runtime.end_step()
```

Block mode gives you control over the block pattern regex, pool sizing, watermarks, and step lifecycle. It supports file-backed and SquareQ-backed streaming that layer mode doesn't (layer mode is module-backed only).

## Install

```bash
pip install stagehand
```

Requires Python ≥3.10, PyTorch ≥2.1.0, safetensors ≥0.4.0, psutil ≥5.9.0.

Works without CUDA (pool uses unpinned memory, transfers are synchronous copies). Useful for testing, not useful for actual offloading.

## RamTorch compatibility shim

Drop-in replacement for `ramtorch.helpers`. Lets RamTorch users (e.g. SimpleTuner) switch to Stagehand with a one-line import change:

```python
# Before:
from ramtorch.helpers import replace_linear_with_ramtorch, move_model_to_device

# After:
from stagehand.compat.ramtorch import replace_linear_with_ramtorch, move_model_to_device
```

The shim calls `stagehand.layer()` internally and sets `is_ramtorch = True` on managed modules/params so downstream `getattr(param, "is_ramtorch", False)` checks (quantization skip, DDP ignore, device move skip) continue to work.

`move_model_to_device()` becomes a no-op when Stagehand is active. `reattach_is_ramtorch_flags()` restores flags after `torch.save`/`torch.load`.

## Internals

### Pool (`PinnedPool`)

Fixed-size pool of page-locked CPU memory slabs. All slabs allocated at init — no runtime allocation. Acquisition blocks on a condition variable if no slabs are free (logs a warning at 100ms). Slabs are 512-byte aligned for DMA.

### Transfer engine (`AsyncTransferEngine`)

Dedicated high-priority CUDA stream (`priority=-1`). Backpressure via `max_inflight` (default: 2). Each H2D/D2H copy records a CUDA event for synchronization. The default stream waits on the copy event before using the data.

### Scheduler (`StagehandScheduler`)

`before_block(block_id)`:
- If layer is already on GPU → prefetch hit
- If layer is mid-transfer → wait for completion (stall)
- Otherwise → initiate load (stall + miss)
- Then: prefetch next *k* layers, run eviction if above watermark

Eviction scoring: `score = distance_to_next_use * size_bytes`. Blocks in the prefetch window are protected from eviction. Eviction runs until VRAM drops below the low watermark.

### Residency state machine

```
UNLOADED → HOST_STAGED → PREFETCHING → GPU_READY → EVICTING → HOST_STAGED
                                                  → GPU_FREEING → UNLOADED
```

Every transition is validated. Invalid transitions raise `InvalidStateTransitionError`. Refcounted — a block with `refcount > 0` is never evicted.

### Telemetry

Per-step metrics written to JSONL: H2D/D2H bytes, stall time/count, eviction count, prefetch hit/miss, VRAM usage, pool utilization, NaN/Inf counts. Rolling 100-step window for aggregate stats (hit rate, mean stall, VRAM trend).

### Numeric guards (block mode only)

Optional `strict_bf16` checking, NaN/Inf detection on block outputs, dtype promotion detection. Disabled in layer mode.

## Shutdown

```python
runtime = model._stagehand_layer_runtime

# Full shutdown — releases pool, closes telemetry, removes hooks
runtime.shutdown()

# Or keep pool for reuse with another model
pool = runtime.shutdown_keep_pool()
model2 = stagehand.layer(other_model, pool=pool)
```

Double `shutdown()` is safe (idempotent).

## Tests

204 tests (169 CPU-only + 35 GPU stress tests). Run with:

```bash
pip install -e ".[dev]"
pytest tests/ -x -q
```

**CPU tests** (run anywhere): pool allocation/release, all 6 residency state transitions (and all invalid ones), registry build/validate/freeze, scheduler prefetch/eviction/stall, layer discovery/trace/rebuild/auto-step, compat shim API + functional correctness, numeric guards, budget watermarks, telemetry recording. Functional correctness tests verify forward output matches the unwrapped model with `atol=1e-5` on both trace and scheduled passes.

**GPU stress tests** (require CUDA, auto-skipped otherwise): real H2D/D2H async transfers, VRAM budgeting with eviction under pressure, gradient survival across evict→reload cycles, OffloadedAdamW state placement, bf16 dtype preservation through D2H→H2D round-trips, training convergence matching vanilla AdamW, RamTorch compat shim training, no-VRAM-leak stress (50 steps), pool reuse across models, and large model (80MB, 40 layers) completion with telemetry verification.

## Limitations

- **Layer mode is module-backed only**. No file-backed or SquareQ streaming. Every eviction round-trips weights through CPU RAM.
- **No multi-GPU support**. Single device only.
- **Prefetch policy is static**. Fixed lookahead window, no adaptive prediction.
- **Gradient accumulation + tight VRAM budget**. When eviction happens during backward, PyTorch's `AccumulateGrad` node runs after the backward post-hook — so eviction moves `param.grad` to CPU before `AccumulateGrad` can accumulate on GPU, causing a device mismatch. Workaround: use a generous VRAM budget for gradient accumulation so no eviction occurs during backward. Single forward+backward per step (the common case) works fine with any budget.
- **No gradient checkpointing integration**. Stagehand's backward hooks coexist with PyTorch's autograd but don't coordinate with `torch.utils.checkpoint`.
- **Pool sizing is fragile**. If the largest layer doesn't fit in one slab, init fails with `StagehandOOMError`. Auto-sizing in layer mode handles this, but block mode requires manual slab sizing.
- **Alpha quality**. Used in [Serenity](https://github.com/CodeAlexx/Serenity) for diffusion model training. The API surface is small but not battle-tested.

## License

MIT
