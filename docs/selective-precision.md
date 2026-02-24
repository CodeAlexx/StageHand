# Selective Precision: Per-Block BF16/INT8 Routing Based on Gradient Sensitivity

When Stagehand streams transformer blocks through GPU one at a time, some blocks are sensitive to quantization error (high gradient activity, steep loss curvature) while others tolerate INT8 with no training quality impact. Selective Precision monitors per-block gradient statistics and dynamically assigns each block to BF16 (via Stagehand) or INT8 (via SquareQ), maximizing bandwidth savings without degrading convergence.

**Status**: v0, integrated in [Serenity](https://github.com/CodeAlexx/Serenity). Operates above Stagehand and SquareQ — does not modify either runtime's internals. Consumed by [Conductor](conductor.md)'s PolicyBridge.

## The Problem

Consider a 48-block transformer trained with Stagehand block-swap on a 24GB GPU. All blocks stream as BF16:

| Block Position | Gradient Activity | Quantization Error | Needs BF16? |
|----------------|------------------|--------------------|-------------|
| Blocks 0-5 (early) | Low | Low (0.01-0.02) | No — INT8 is fine |
| Blocks 20-30 (mid) | Moderate | Moderate (0.03-0.04) | Mixed |
| Blocks 44-47 (late) | High | High (0.05-0.08) | Yes — bf16 required |

Streaming all 48 blocks as BF16 wastes PCIe bandwidth on the ~60% of blocks that would train identically at INT8. SquareQ transfers INT8 blocks at half the bytes — but blindly quantizing everything hurts the sensitive blocks that need full precision.

## How It Works

Selective Precision has three stages:

### 1. Gradient Collection

After each backward pass, `collect_grad_stats()` iterates through registered blocks and measures:

- **L2 norm** of all gradient tensors in the block
- **Max absolute gradient** value
- **Gradient variance**
- **Relative magnitude** — this block's L2 norm divided by the mean across all blocks

These stats are accumulated in a sliding window (default: 5 steps).

### 2. Sensitivity Scoring

At each update interval (default: every 10 steps), the policy averages gradient stats over the history window and computes a sensitivity score per block:

```
sensitivity = grad_weight * grad_score + error_weight * error_score
```

Where:
- `grad_score = min(relative_magnitude / grad_sensitivity_threshold, 1.0)` — blocks with above-average gradient activity score higher
- `error_score = min(quant_error / quant_error_threshold, 1.0)` — blocks with high calibration error score higher (optional, from calibration pass)
- Default weights: `grad_weight=0.7`, `error_weight=0.3`

The score is clamped to [0.0, 1.0] where 0 = safe to quantize, 1 = keep bf16.

### 3. Hysteresis Decision

To prevent precision flickering, the decision logic applies:

- **Threshold separation**: bf16 threshold (0.6) is above int8 threshold (0.3), creating an ambiguous zone where blocks keep their current assignment
- **Hysteresis margin**: A block currently at bf16 must drop below `int8_threshold - hysteresis_margin` to switch to int8
- **Cooldown period**: Blocks cannot change precision within `min_steps_between_switches` (default: 20) steps of their last change

```
Currently bf16 → switch to int8 only if sensitivity < 0.2 (threshold 0.3 - margin 0.1)
Currently int8 → switch to bf16 only if sensitivity ≥ 0.6
In between → keep current assignment
```

## Quantization Error Calibration

An optional calibration pass measures per-block output error between BF16 and INT8 weights before training begins:

1. For each block, run a small number of forward passes (default: 4) with real activations
2. Compare BF16 output vs INT8 output: `relative_error = ||out_bf16 - out_int8|| / ||out_bf16||`
3. Average across samples to get a stable error estimate

Calibration results are cached to disk with a fingerprint of model identity + quant config + block layout, so they only run once per configuration.

The calibration error feeds into the sensitivity score via the `error_weight` component. Without calibration, the policy relies on gradient statistics alone.

## Three Modes

| Mode | Behavior |
|------|----------|
| **off** | All blocks BF16. Equivalent to no Selective Precision. |
| **static** | Use `force_bf16_blocks` / `force_int8_blocks` lists only. No dynamic scoring. |
| **dynamic** | Full gradient-based scoring with hysteresis. Manual overrides still take priority. |

Manual overrides (`force_bf16_blocks`, `force_int8_blocks`) take absolute priority in all modes. Use these for blocks you know empirically need a specific precision.

## Integration Point

Selective Precision is wired into `StagehandStrategy` in Serenity. Conductor's PolicyBridge reads the block precision hints and includes them in `RuntimeLimitHints`:

```python
# In training config JSON:
{
  "memory": {
    "stagehand": {
      "selective_precision": {
        "enabled": true,
        "mode": "dynamic",
        "warmup_steps": 10,
        "update_interval_steps": 10,
        "bf16_threshold": 0.6,
        "int8_threshold": 0.3
      }
    }
  }
}
```

The lifecycle in the training loop:

```
begin_step(step)
  forward pass
  backward pass
  collect_block_grad_stats()   ← SP collects gradient norms
  compute_hints(step)          ← SP updates precision assignments (every N steps)
  optimizer.step()
end_step()
```

## Configuration

```json
{
  "selective_precision": {
    "enabled": true,
    "mode": "dynamic",
    "bf16_threshold": 0.6,
    "int8_threshold": 0.3,
    "ambiguous_default": "bf16",
    "hysteresis_margin": 0.1,
    "grad_weight": 0.7,
    "error_weight": 0.3,
    "grad_sensitivity_threshold": 2.0,
    "run_calibration": true,
    "calibration_samples": 4,
    "quant_error_threshold": 0.05,
    "warmup_steps": 10,
    "history_window": 5,
    "update_interval_steps": 10,
    "min_steps_between_switches": 20,
    "force_bf16_blocks": [],
    "force_int8_blocks": [],
    "log_decisions": true,
    "telemetry_enabled": true,
    "telemetry_file": "selective_precision_telemetry.jsonl"
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"dynamic"` | `"off"`, `"static"`, or `"dynamic"` |
| `bf16_threshold` | 0.6 | Sensitivity above this → assign bf16 |
| `int8_threshold` | 0.3 | Sensitivity below this → assign int8 |
| `ambiguous_default` | `"bf16"` | Default for scores between thresholds |
| `hysteresis_margin` | 0.1 | Extra margin before switching away from current precision |
| `grad_weight` | 0.7 | Weight of gradient score in sensitivity |
| `error_weight` | 0.3 | Weight of calibration error in sensitivity |
| `grad_sensitivity_threshold` | 2.0 | Relative magnitude above this = max grad score |
| `run_calibration` | true | Run quantization error calibration before training |
| `calibration_samples` | 4 | Number of forward passes for calibration |
| `quant_error_threshold` | 0.05 | Relative error above this = max error score |
| `warmup_steps` | 10 | Steps of gradient collection before first dynamic assignment |
| `history_window` | 5 | Number of recent backward passes to average |
| `update_interval_steps` | 10 | Steps between hint recomputation |
| `min_steps_between_switches` | 20 | Cooldown before a block can change precision again |
| `force_bf16_blocks` | [] | Block IDs forced to bf16 regardless of score |
| `force_int8_blocks` | [] | Block IDs forced to int8 regardless of score |
| `log_decisions` | true | Log precision changes to Python logger |
| `telemetry_enabled` | true | Write per-update metrics to JSONL |

## Telemetry

Selective Precision writes per-update metrics to `selective_precision_telemetry.jsonl`:

```json
{
  "step_id": 50,
  "timestamp": 1234567.89,
  "blocks_bf16": 19,
  "blocks_int8": 29,
  "mean_sensitivity": 0.34,
  "max_sensitivity": 0.87,
  "min_sensitivity": 0.04,
  "precision_changes": 2,
  "estimated_bandwidth_saving_pct": 30.2,
  "block_details": {}
}
```

`estimated_bandwidth_saving_pct` is a rough estimate — each INT8 block transfers half the bytes of a BF16 block, so 60% INT8 blocks ≈ 30% bandwidth saving.

## Architecture

```
serenity/memory/selective_precision/
    __init__.py           # Package exports
    config.py             # SelectivePrecisionConfig dataclass
    sensitivity.py        # BlockGradStats, BlockPrecisionHint, collect_grad_stats,
                          #   compute_sensitivity_score, decide_with_hysteresis
    calibration.py        # run_calibration, CalibrationResult, disk cache
    policy.py             # SelectivePrecisionPolicy (main orchestrator)
    telemetry.py          # PrecisionStepMetrics, PrecisionTelemetry (JSONL)
```

### Relationship to Conductor

Conductor's `HeuristicPolicyBridgeV0` composes with `SelectivePrecisionPolicy`:

```
SelectivePrecisionPolicy
    ├── Gradient collection  ← fed by training loop after backward
    ├── Sensitivity scoring  ← averaged over history window
    └── Hint emission        ← block_id → BlockPrecisionHint

Conductor PolicyBridge
    ├── Reads SP hints       ← block_precision_map in RuntimeLimitHints
    ├── Phase rules          ← VRAM pressure, optimizer protection
    └── Pushes to adapters   ← Stagehand / SquareQ choose transfer path
```

Selective Precision decides *which* blocks get which precision. Conductor decides *when* and *how fast* they transfer. They compose without coupling.

## v0 Limitations

- **Gradient-only signal**: Sensitivity scoring relies on gradient norms. Blocks with small gradients but high activation sensitivity may be misclassified.
- **No per-layer granularity**: Operates at the transformer block level (attention + MLP + norms together). Individual layers within a block always share the same precision.
- **Calibration requires forward hooks**: The `get_block_input` callback needs real activations. During Stagehand block-swap, only one block is on GPU at a time — calibration must run blocks sequentially.
- **No FP8 support**: Only bf16 and int8 are supported in v0. FP8 (E4M3/E5M2) is a natural future extension.
- **Fixed scoring weights**: `grad_weight` and `error_weight` are constants. Adaptive weighting based on training phase could improve decisions.
- **No loss feedback**: Decisions don't observe training loss. A block switched to int8 that causes loss spikes won't be detected until gradient statistics change.

## See Also

- [Conductor](conductor.md) — resource arbitration that consumes Selective Precision hints
- [Activation Stagehand](activation-stagehand.md) — spilling autograd activations to pinned CPU memory
- [Case Study: LTX-2 bf16 on 24GB](case-study-ltx2-bf16.md) — two-stage training using both Weight and Activation Stagehand
