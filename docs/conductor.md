# Conductor: Resource Arbitration Across Memory Runtimes

When multiple memory runtimes (weight Stagehand, Activation Stagehand, SquareQ) run simultaneously during training, they compete for the same VRAM headroom, pinned CPU budget, and PCIe bandwidth. Conductor is a thin coordination layer that arbitrates these shared resources without rewriting any runtime internals.

**Status**: v0, integrated in [Serenity](https://github.com/CodeAlexx/Serenity). Operates above Stagehand — does not modify Stagehand's core logic. No standalone library integration yet.

## The Problem

Consider LTX-2 two-stage video training on a 24GB GPU. During stage 2 (transformer training), three runtimes are active simultaneously:

| Runtime | Uses VRAM | Uses Pinned CPU | PCIe Direction |
|---------|-----------|-----------------|----------------|
| Weight Stagehand | Prefetch window (1-3 blocks) | 8 GB pinned pool | H2D (load blocks) |
| Activation Stagehand | Spill/restore activations | Slab pool (~500 MB) | D2H (spill) + H2D (restore) |
| SquareQ | INT8 dequant staging | Slab backing (~256 MB) | H2D (dequant) |

Without coordination:
- All three issue PCIe transfers concurrently, saturating bandwidth and increasing stall time
- Weight prefetch during optimizer step wastes VRAM (optimizer needs that headroom for state)
- Activation spills during backward compete with weight loads for the same PCIe lanes

## How Conductor Works

Conductor does three things:

### 1. Phase Broadcasting

The training step is divided into phases: `STEP_BEGIN → FORWARD → BACKWARD → OPTIMIZER → STEP_END`. Conductor broadcasts phase transitions to all registered runtime adapters, so each runtime knows what's happening.

### 2. Budget Arbitration

A `BudgetManager` tracks VRAM headroom and pinned CPU reservations with typed modes:

| Mode | Behavior |
|------|----------|
| **HARD** | Full grant or deny — no partial |
| **SOFT** | Partial grants OK (get what's available) |
| **BURST** | Can exceed soft cap up to hard cap |
| **FLOOR** | Guaranteed minimum — full grant or deny |
| **CEILING** | Upper bound hint |

Reservations can be phase-scoped (auto-released when a phase ends) and carry priority levels (CRITICAL > REQUIRED > SPECULATIVE > BACKGROUND).

### 3. Transfer Slot Management

A `TransferManager` maintains separate H2D and D2H slot pools. Runtimes acquire tokens before issuing PCIe transfers. If all slots are taken, the request gets a typed denial reason (e.g., `H2D_SLOTS_EXHAUSTED`, `PHASE_RULE_SUPPRESSED_SPECULATIVE`).

## Policy Rules (v0)

The `HeuristicPolicyBridgeV0` computes `RuntimeLimitHints` at every phase boundary and pushes them to all adapters. Three rules:

**Rule 1 — Backward pressure**: When VRAM pressure exceeds 80% during BACKWARD, suppress speculative prefetch and reduce the prefetch window to 1. This frees VRAM for activation accumulation.

**Rule 2 — Optimizer protection**: During OPTIMIZER phase, suppress all speculative work and limit H2D to 1 slot. The optimizer needs headroom for state updates — prefetching blocks into VRAM during this phase is wasted work.

**Rule 3 — Contention reduction**: When all transfer slots are consistently full (>3 consecutive checks), progressively reduce the prefetch window. This backs off PCIe pressure.

**Monotonicity guarantee**: Within a single step, hints only tighten — `prefetch_window` is non-increasing and `suppress_speculative` never reverts from True to False. Resets at `STEP_BEGIN`.

## What Gets Tuned

Conductor pushes hints to runtime adapters, which translate them into internal knob writes:

| Hint | Weight Stagehand | Activation Stagehand | SquareQ |
|------|-----------------|---------------------|---------|
| `max_inflight_h2d` | `engine._max_inflight` | `transfer._max_h2d` | N/A |
| `max_inflight_d2h` | N/A | `transfer._max_d2h` | N/A |
| `prefetch_window_cap` | `scheduler.policy.prefetch_window` | N/A | N/A |
| `suppress_speculative` | sets `prefetch_window = 1` | sets `_max_d2h = 0` in OPTIMIZER | N/A |

All writes are to plain instance variables on the training thread — same thread that calls `before_block()` / `_wait_for_slot()`. No synchronization issues.

Adapters save original values on `attach()` and restore them on `detach()`.

## Zero-Cost Disable

```python
from serenity.memory.conductor import ConductorConfig, ConductorRuntime

# Disabled — no sub-objects allocated, all methods return immediately
rt = ConductorRuntime(ConductorConfig(enabled=False))
rt.begin_step(0)    # no-op
rt.enter_forward()  # no-op
rt.shutdown()       # no-op
```

When `enabled=False`, the constructor returns after storing `_enabled=False`. No BudgetManager, no TransferManager, no PhaseCoordinator — nothing is allocated.

## Integration Point

Conductor is wired into `StagehandStrategy.forward_context()` in Serenity:

```python
# In training config JSON:
{
  "memory": {
    "stagehand": {
      "conductor_config": {
        "enabled": true,
        "vram_soft_cap_mb": 22000,
        "vram_hard_cap_mb": 23500,
        "h2d_slots": 2,
        "d2h_slots": 2
      }
    }
  }
}
```

The strategy creates a `ConductorRuntime`, registers adapters for whichever runtimes are active, and calls lifecycle methods at the right points:

```
begin_step(step)
  enter_forward()
    [forward hooks fire — weight stagehand streams blocks]
    [loss.backward() — backward hooks fire]
  enter_backward()
  enter_optimizer()
    [optimizer.step()]
  end_step()
```

## Telemetry

Conductor writes per-step metrics to `conductor_telemetry.jsonl` at configurable intervals:

```json
{
  "step_id": 50,
  "vram_allocated_mb": 21456.3,
  "vram_headroom_mb": 2043.7,
  "pinned_granted_mb": 8704.0,
  "h2d_inflight": 1,
  "d2h_inflight": 0,
  "grant_count": 12,
  "deny_count": 0,
  "partial_count": 2,
  "phase_durations": {"forward": 3.21, "backward": 4.56, "optimizer": 1.89},
  "runtime_snapshots": {
    "stagehand": {"max_inflight": 2, "prefetch_window": 2},
    "activation_stagehand": {"max_d2h": 1, "max_h2d": 1}
  }
}
```

Set `debug_event_trace: true` for fine-grained event logging (phase transitions, reservation requests, slot acquisitions).

## Architecture

```
ConductorRuntime
├── PhaseCoordinator ──→ broadcasts to all adapters
├── BudgetManager ────→ VRAM + pinned CPU reservations
├── TransferManager ──→ H2D / D2H slot pools
├── PolicyBridge ─────→ computes RuntimeLimitHints per phase
├── TelemetryHub ─────→ JSONL metrics + debug events
└── Adapters
    ├── StagehandConductorAdapter ──→ tunes weight prefetch
    ├── ActStagehandConductorAdapter → tunes activation spill/restore
    └── SquareQConductorAdapter ────→ reports pinned usage (no knobs v0)
```

## What Conductor Does NOT Do

- Does not rewrite any runtime internals (Stagehand, Activation Stagehand, SquareQ)
- Does not make scheduling decisions — runtimes still decide when/what to prefetch
- Does not allocate VRAM or pinned memory — only tracks and constrains
- Does not add overhead when disabled (zero-cost gate)
- Does not require all three runtimes — works with any subset
