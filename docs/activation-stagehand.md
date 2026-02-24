# Activation Stagehand: Spilling Autograd Activations to Pinned CPU Memory

Weight Stagehand streams transformer blocks through GPU one at a time. But during forward, PyTorch's autograd saves intermediate tensors (activations) for backward — and these accumulate across ALL blocks, consuming 2-5x more VRAM than a single block's weights.

Activation Stagehand intercepts autograd's `saved_tensors_hooks` to spill activation tensors from GPU to pinned CPU memory when VRAM pressure exceeds a watermark, then restores them on demand during backward.

**Status**: v0, integrated in [Serenity](https://github.com/CodeAlexx/Serenity). Standalone library integration planned for a future Stagehand release.

## The Problem

Consider LTX-2 training with weight Stagehand on a 24GB GPU:

| Component | VRAM |
|-----------|------|
| Current weight block (1 of 48) | ~3 GB |
| Small layers (norms, projections, LoRA) | ~2 GB |
| Optimizer state fragments | ~1 GB |
| **Accumulated activations (all processed blocks)** | **~12-14 GB** |

Weight Stagehand keeps only one block on GPU at a time, but autograd holds activations from *every* block that has run so far. By block 40 of 48, there are 40 blocks' worth of activation tensors on GPU — even with gradient checkpointing reducing per-block activation memory.

The VRAM peak occurs mid-forward when accumulated activations + current weight block + overhead exceed 24GB.

## How It Works

Activation Stagehand uses `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` — a PyTorch API that intercepts every tensor autograd saves during forward.

### Forward (pack)

For each tensor autograd wants to save:

1. **Check VRAM pressure** via `torch.cuda.memory_allocated()` against the high watermark
2. If below watermark → **KEEP** on GPU (zero overhead)
3. If above watermark → **SPILL**: async D2H copy to a pinned CPU buffer, release GPU tensor, return a handle
4. Return either the original tensor (KEPT) or a record ID (SPILLED)

### Backward (unpack)

When autograd needs a saved tensor:

1. If KEPT → return the GPU tensor directly (zero overhead)
2. If SPILLED → **RESTORE**: sync H2D copy from pinned CPU buffer back to GPU, return the restored tensor

### Inline Finalize

The D2H transfer engine has an inflight cap (default: 1) to avoid saturating PCIe bandwidth shared with weight Stagehand. To avoid the cap blocking subsequent spills during forward:

```python
# In _pack(), before starting a new spill:
if not self._transfer.can_spill():
    # Finalize any in-flight spills to free up the slot
    spilling = self._registry.iter_spilling()
    if spilling:
        self._transfer.finalize_all_spills(spilling)
```

This waits for the previous D2H copy to complete before starting the next one. Without this, only 1 tensor per step would be spilled regardless of VRAM pressure.

## Size-Class Pinned Pool

Activation tensors range from <1KB to >200MB. A uniform-slab pool (like weight Stagehand's `PinnedPool`) wastes memory on small tensors and can't handle large ones.

Activation Stagehand uses a **size-class pool** with multiple bucket sizes:

```
Class sizes:   1 MB    4 MB    16 MB   64 MB   256 MB
Slabs/class:   512     2       2       2       2
Total pinned:  520 MB  8 MB    32 MB   128 MB  512 MB  = ~1.2 GB
```

The dominant activation tensor size in LTX-2 training is ~0.9MB (attention intermediates). The 512 slabs at 1MB accommodate all ~420 tensors that are spilled simultaneously during forward. Larger classes handle occasional oversized tensors (projection outputs, cross-attention).

### Per-Class Slab Counts

Slab counts are specified per class to match the actual tensor size distribution:

```python
CPUSpillBackend(
    class_sizes_mb=(1, 4, 16, 64, 256),
    slabs_per_class=(512, 2, 2, 2, 2),   # 512 x 1MB, 2 x 4MB, etc.
)
```

A single int gives uniform counts across all classes. A tuple parallel to `class_sizes_mb` gives per-class control.

### Allocation Strategy

1. Find the smallest class whose slab size >= needed bytes
2. If a free slab exists → return it (pool hit)
3. If not → try the next larger class
4. If all classes exhausted → fallback: allocate an unpinned CPU tensor (pool miss)

Released buffers return to their original size class via data pointer matching.

## Real Results: LTX-2 on 3090 Ti

50 training steps, LoRA rank 32, 512px 17-frame video, watermarks 16000/12000 MB:

### Before Activation Stagehand

| Metric | Value |
|--------|-------|
| VRAM peak (allocated) | 19.4 GB |
| Activations kept on GPU | 636 |
| Activations spilled | 0 |
| Step time | 9.3 s |

### After (with inline finalize + sized pool)

| Metric | Value |
|--------|-------|
| VRAM peak (allocated) | 16.9 GB |
| Activations kept on GPU | 214 |
| Activations spilled per step | 422 |
| Spill volume per step | 376 MB |
| Pool hit rate | 98%+ |
| Step time | 10.0 s |

**2.5 GB VRAM reduction** with 8% speed overhead. The VRAM headroom freed by spilling activations prevents OOM on tighter VRAM budgets and leaves room for larger batch sizes or resolutions.

### Telemetry (per-step JSONL)

```json
{
  "step": 12,
  "activations_saved": 636,
  "activations_kept": 214,
  "activations_spilled": 422,
  "activations_restored": 422,
  "spill_bytes": 394526720,
  "restore_bytes": 394526720,
  "stall_time_ms": 0.0,
  "stall_count": 0,
  "pool_hits": 420,
  "pool_misses": 2,
  "vram_peak_mb": 16894.3
}
```

## Architecture

```
serenity/memory/activation/
    __init__.py           # Package exports
    config.py             # ActivationStagehandConfig dataclass
    registry.py           # ActivationState, ActivationRecord, ActivationRegistry
    backend_cpu.py        # CPUSpillBackend (size-class pinned pool)
    transfer.py           # ActivationTransferEngine (async D2H/H2D on dedicated CUDA streams)
    hooks.py              # ActivationHookContext (saved_tensors_hooks pack/unpack)
    budget.py             # ActivationBudgetManager (watermark tracking)
    telemetry.py          # ActivationStepMetrics + ActivationTelemetry (JSONL)
    scheduler.py          # ActivationScheduler (v0 heuristic)
    runtime.py            # ActivationRuntime (lifecycle orchestrator)
    checkpoint.py         # CheckpointCoordinator (recompute preference flag)
```

### Integration with Weight Stagehand

Activation hooks nest inside weight Stagehand's `forward_context()`:

```python
# In StagehandStrategy.forward_context():
runtime.begin_step(step)
activation_runtime.step_begin(step)

with runtime.managed_forward():
    with runtime.managed_backward():
        with activation_runtime.managed_forward():
            yield  # training loop runs here

activation_runtime.step_end()
runtime.end_step()
```

Both systems share PCIe bandwidth. Activation Stagehand uses separate CUDA streams with conservative inflight caps (1 D2H + 1 H2D) to avoid starving weight prefetch.

## Configuration

```json
{
  "memory": {
    "strategy": "stagehand",
    "stagehand": {
      "activation_stagehand": {
        "enabled": true,
        "vram_high_watermark_mb": 16000,
        "vram_low_watermark_mb": 12000,
        "pinned_pool_classes_mb": [1, 4, 16, 64, 256],
        "slabs_per_class": [512, 2, 2, 2, 2],
        "max_inflight_d2h": 1,
        "max_inflight_h2d": 1,
        "debug_checksums": false,
        "telemetry_enabled": true,
        "telemetry_file": "activation_telemetry.jsonl"
      }
    }
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vram_high_watermark_mb` | 20000 | Start spilling above this |
| `vram_low_watermark_mb` | 16000 | Stop spilling below this |
| `pinned_pool_classes_mb` | (1,4,16,64,256) | Size-class bucket sizes |
| `slabs_per_class` | (512,2,2,2,2) | Slabs per size class |
| `max_inflight_d2h` | 1 | Max concurrent GPU→CPU copies |
| `max_inflight_h2d` | 1 | Max concurrent CPU→GPU copies |
| `debug_checksums` | false | CRC32 checksums on spill/restore for validation |
| `telemetry_enabled` | true | Per-step JSONL telemetry |
| `recompute_threshold_bytes` | 0 | Prefer recompute over spill for tensors below this size (v0: unused) |

## v0 Limitations

- **Reactive scheduler**: Spills on watermark pressure, restores on demand. No predictive prefetch using backward execution order.
- **No gradient checkpointing coordination**: Works alongside `torch.utils.checkpoint` but doesn't know which activations will be recomputed.
- **Fixed pool at init**: No dynamic growth. If the pool is too small, falls back to unpinned allocations.
- **Single-step scope**: All activation records are cleared at step boundaries. No cross-step persistence.
- **No conductor integration**: Doesn't coordinate with a global PCIe bandwidth arbiter. Inflight caps are the only backpressure mechanism.

## Relationship to Weight Stagehand

| Concern | Weight Stagehand | Activation Stagehand |
|---------|-----------------|---------------------|
| **What it manages** | Model weights (parameters) | Autograd saved tensors (activations) |
| **Granularity** | Per-block (all params in a transformer block) | Per-tensor (individual activation tensors) |
| **Pool type** | Uniform-slab PinnedPool | Size-class CPUSpillBackend |
| **Transfer trigger** | Scheduler (prefetch + eviction) | Watermark pressure (reactive) |
| **CUDA streams** | Shared high-priority stream | Separate dedicated streams |
| **Lifecycle** | Block enter/exit hooks | `saved_tensors_hooks` pack/unpack |
| **Persistence** | Weights survive across steps | Activations cleared each step |
