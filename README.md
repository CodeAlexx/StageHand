# Stagehand

GPU memory orchestrator for PyTorch. Streams model weights between CPU and GPU so models larger than VRAM can run without quantization.

**Status**: Alpha (`0.1.0`). API may change. Used in [Serenity](https://github.com/CodeAlexx/Serenity) and [ai-toolkit](https://github.com/ostris/ai-toolkit) for diffusion model training.

**Case studies** (these use Stagehand within [Serenity](https://github.com/CodeAlexx/Serenity)):
- [LTX-2 19B + Gemma 3 12B trained in full bf16 on a single 24GB GPU](docs/case-study-ltx2-bf16.md) — 31.9B total parameters, no quantization, 9.5s/step
- [Flux 2 Dev 12B + Mistral 3 24B with SquareQ INT8 on a single 24GB GPU](docs/case-study-flux2dev-squareq.md) — 36B total parameters, INT8 frozen weights, 6 GB VRAM steady state
- [Flux 1 Dev 12B LoRA at 2048x2048 on 4.15 GB VRAM](docs/case-study-flux-2048-lora.md) — full bf16, no quantization, 90% of layers offloaded to CPU
- [Porting the Conductor to LTX-Desktop](docs/case-study-ltx-desktop-conductor.md) — LTX-2 19B LoRA on 24 GB, and the `use_reentrant` gradient bug with dataclass block interfaces

**Third-party integrations**:
- [ai-toolkit](https://github.com/ostris/ai-toolkit) — Kohya-style block swap for training + Turbo engine for sampling. See [integration docs](docs/ai-toolkit-integration.md).

## Docs

### Stagehand library

- [Residency Protection](docs/residency-protection.md) — `keep_resident()`, `reserve_for_resident()`, and guest model scoping for multi-model VRAM management

### Serenity integration

These features are built on top of Stagehand but implemented in [Serenity](https://github.com/CodeAlexx/Serenity), not in this library:

- [Activation Stagehand](docs/activation-stagehand.md) — spilling autograd activations to pinned CPU memory
- [Conductor](docs/conductor.md) — resource arbitration across weight Stagehand, Activation Stagehand, and SquareQ
- [Per-Layer & Per-Component Learning Rates](docs/per-layer-lr.md) — 2D LR grid (component type x block depth) for LoRA and full-finetune training
- [Selective Precision](docs/selective-precision.md) — per-block BF16/INT8 routing based on gradient sensitivity

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

## Turbo engine (Rust)

For sampling/inference, Stagehand includes an optional Rust-based engine (`TurboConductor`) that manages GPU slot pools with dedicated CUDA streams. It's ~10x faster than the Python path because it eliminates Python overhead from the hot loop.

```python
from stagehand.turbo import TurboConductor, is_available

if is_available():
    turbo = TurboConductor(
        model=transformer,
        block_pattern=r"(transformer_blocks|single_transformer_blocks)\.\d+$",
        dtype=torch.bfloat16,
        device=0,
        vram_budget_gb=18.0,
    )
    turbo.prepare()

    # Per-block hooks call turbo.before_block(i) / turbo.after_block(i)
    # Turbo handles prefetch, eviction, and stream synchronization in Rust
```

**Requirements**: `pip install stagehand[turbo]` (builds the Rust crate via maturin). Falls back to Python path if unavailable.

**How it works**: Pre-allocates a fixed pool of GPU memory slots sized to the largest block. Dedicated CUDA streams handle H2D copies. `before_block()` ensures the block is resident (prefetching if needed), `after_block()` marks it reclaimable. All synchronization uses CUDA events, not Python locks.

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

`move_model_to_device()` becomes a no-op when Stagehand is active (and skips `is_ramtorch` params in fallback mode). `reattach_is_ramtorch_flags()` restores flags after `torch.save`/`torch.load`. Hook helpers (`register_ramtorch_grad_hook`, `register_ramtorch_post_accumulate_grad_hook`, `add_custom_hooks`) are also provided for full API parity.

**Limitation**: Stagehand wraps modules with hooks instead of replacing them with a different class. Code that uses `isinstance(module, ramtorch.Linear)` to identify managed modules will get `False` — use `getattr(module, "is_ramtorch", False)` instead.

## VMM backend (experimental)

On Ampere+ GPUs (SM 8.0+), Stagehand can use CUDA Virtual Memory Management for zero-overhead weight residency instead of block-swap. This is provided by the [stagehand-vmm](https://github.com/CodeAlexx/stagehand-vmm) Rust crate.

**How it works**: Instead of copying weights between CPU and GPU via DMA, VMM maps physical VRAM pages directly into the model's virtual address space. The hot path (checking if a block is already resident) takes **250 ns** — an atomic load + refcount increment, no CUDA calls, no mutex. Getting a PyTorch tensor from a resident region takes **1.4 us** via DLPack zero-copy.

**Two-layer architecture**:
- **Layer 1 (RAM)**: Safetensors file mapped with `mmap(MAP_NORESERVE)`. OS manages pages as cache — zero committed RAM. Pages read from disk on demand, reclaimable instantly under pressure.
- **Layer 2 (VRAM)**: CUDA VMM maps physical VRAM on demand (`cuMemCreate` + `cuMemMap`). Eviction via `cuMemUnmap` + `cuMemRelease`. CAS-based protocol prevents races with concurrent access.

**Dual prefetch**: When prefetching a block, both layers warm simultaneously — `madvise(MADV_WILLNEED)` pre-faults file pages while the VMM async worker maps VRAM. By the time the block executes, both source and destination are ready.

**Fallback**: VMM is optional. If `stagehand-vmm` isn't installed, the GPU is pre-Ampere, or a region is watermarked (VRAM ceiling exceeded), Stagehand falls back to the existing block-swap path automatically. No code changes needed.

```python
# VMM activates automatically in inference mode on Ampere+ GPUs
runtime = StagehandRuntime(model, cfg, inference_mode=True)
runtime.convert_registry_to_file_backed_sharded("/path/to/model")
runtime.vmm_register_model("flux-dev", safetensors_path="/path/to/model.safetensors")

with runtime.managed_forward():
    output = model(input)
```

**Benchmarks** (RTX 3090, 24GB):
| Operation | Latency |
|-----------|---------|
| ensure_resident (hot path) | 250 ns median |
| ensure_resident + as_tensor | 1.4 us median |
| Eviction + re-map (1.5GB region) | 4.2 ms median |

**Requirements**: Ampere+ GPU (RTX 30xx, A100, etc.), Linux, `pip install stagehand-vmm`.

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

228 tests (193 CPU-only + 35 GPU stress tests). Run with:

```bash
pip install -e ".[dev]"
pytest tests/ -x -q
```

**CPU tests** (run anywhere): pool allocation/release, all 6 residency state transitions (and all invalid ones), registry build/validate/freeze, scheduler prefetch/eviction/stall, layer discovery/trace/rebuild/auto-step, compat shim API + functional correctness, numeric guards, budget watermarks, telemetry recording, residency protection (`keep_resident`, `reserve_for_resident`, `as_guest` scoping). Functional correctness tests verify forward output matches the unwrapped model with `atol=1e-5` on both trace and scheduled passes.

**GPU stress tests** (require CUDA, auto-skipped otherwise): real H2D/D2H async transfers, VRAM budgeting with eviction under pressure, gradient survival across evict→reload cycles, OffloadedAdamW state placement, bf16 dtype preservation through D2H→H2D round-trips, training convergence matching vanilla AdamW, RamTorch compat shim training, no-VRAM-leak stress (50 steps), pool reuse across models, and large model (80MB, 40 layers) completion with telemetry verification.

## Limitations

- **Layer mode is module-backed only**. No file-backed or SquareQ streaming. Every eviction round-trips weights through CPU RAM.
- **No multi-GPU support**. Single device only.
- **Prefetch policy is static**. Fixed lookahead window, no adaptive prediction.
- **Gradient accumulation + tight VRAM budget**. When eviction happens during backward, PyTorch's `AccumulateGrad` node runs after the backward post-hook — so eviction moves `param.grad` to CPU before `AccumulateGrad` can accumulate on GPU, causing a device mismatch. Workaround: use a generous VRAM budget for gradient accumulation so no eviction occurs during backward. Single forward+backward per step (the common case) works fine with any budget.
- **No gradient checkpointing integration**. Stagehand's backward hooks coexist with PyTorch's autograd but don't coordinate with `torch.utils.checkpoint`.
- **Pool sizing is fragile**. If the largest layer doesn't fit in one slab, init fails with `StagehandOOMError`. Auto-sizing in layer mode handles this, but block mode requires manual slab sizing.
- **Alpha quality**. Used in [Serenity](https://github.com/CodeAlexx/Serenity) and [ai-toolkit](https://github.com/ostris/ai-toolkit) for diffusion model training. The API surface is small but not battle-tested.

## License

MIT
