# Per-Layer & Per-Component Learning Rate Scheduling

Fine-grained learning rate control for LoRA and full-finetune training. Instead of one flat LR for all parameters, each trainable parameter gets:

```
lr = component_lr[type] * depth_fn(block_idx, num_blocks)
```

This creates a 2D grid: **component type** (self_attn, cross_attn, ffn, norm) x **block depth** (0 to N-1).

**Status**: Implemented in [Serenity](https://github.com/CodeAlexx/Serenity). Works with all supported model families. No changes to Stagehand, Activation Stagehand, or the training loop itself -- only optimizer param group construction.

## Why

Standard LoRA training uses one learning rate for every parameter. But:

- **Shallow blocks** see more general features (edges, colors). They benefit from higher LR to adapt quickly.
- **Deep blocks** see more abstract/semantic features. They're more sensitive and can destabilize with high LR.
- **Self-attention** (Q/K/V projections) directly controls what the model attends to. Usually needs a higher LR.
- **Cross-attention** controls how the model processes text conditioning. Often benefits from a lower LR to avoid destroying text alignment.
- **FFN layers** are the "memory" of each block. Moderate LR works well.
- **Norm layers** are rarely LoRA-targeted, but when they are, they need very low LR.

The per-layer LR system lets you express all of this in one config block.

## Config

Add a `per_layer_lr` block to your training JSON config:

```json
{
  "learning_rate": 0.0004,
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "cross_attn": 0.0002,
      "ffn": 0.0003,
      "norm": 0.00005
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.3,
    "peak_position": 0.5,
    "default_lr": 0.0003
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Master switch |
| `component_lr` | dict | see above | Base LR per component type before depth scaling |
| `depth_strategy` | string | `"linear"` | Depth function: `linear`, `cosine`, `inverted_u`, `flat` |
| `min_depth_factor` | float | `0.3` | Minimum depth scaling factor (0.0 - 1.0) |
| `peak_position` | float | `0.5` | Peak position for `inverted_u` strategy (0.0 - 1.0) |
| `default_lr` | float | `0.0003` | LR for unmatched parameters and optimizer base LR |
| `log_grad_norms` | bool | `false` | Reserved for future gradient norm logging |

When `per_layer_lr.enabled` is false or absent, training uses the flat `learning_rate` as before. No behavior change.

## Depth Strategies

All strategies produce a factor between `min_depth_factor` and `1.0`. The final LR is `component_lr[type] * factor`.

### `linear` (default)

Linearly decays from 1.0 at block 0 to `min_depth_factor` at block N-1.

```
factor(idx) = 1.0 - (idx / (N-1)) * (1.0 - min_depth_factor)
```

```
Block:  0     5     10    15    19
Factor: 1.00  0.82  0.63  0.45  0.30
```

Best for: general-purpose training where shallow blocks should learn faster.

### `cosine`

Cosine decay from 1.0 to `min_depth_factor`. More gradual at the start, steeper in the middle.

```
factor(idx) = min_factor + (1 - min_factor) * 0.5 * (1 + cos(pi * idx/(N-1)))
```

```
Block:  0     5     10    15    19
Factor: 1.00  0.90  0.65  0.40  0.30
```

Best for: smoother transitions when you don't want a sharp LR cliff.

### `inverted_u`

Quadratic peak at `peak_position`, decaying to `min_depth_factor` at both ends.

```
Block:  0     5     10    15    19
Factor: 0.30  0.75  1.00  0.75  0.30   (peak_position=0.5)
```

Best for: when middle blocks are most important (common in image models where mid-level features matter most for style).

### `flat`

No depth variation. Factor is always 1.0. Only component-level LR differences apply.

```
Block:  0     5     10    15    19
Factor: 1.00  1.00  1.00  1.00  1.00
```

Best for: when you only want per-component LR without depth scaling.

## Component Patterns

Each trainable parameter is classified by matching its name against these patterns:

| Component | Patterns | Typical Role |
|-----------|----------|------|
| `self_attn` | `attn.to_q/k/v/out`, `attn1.to_*`, `attention.to_*`, `to_q.lora`/`to_k.lora`/`to_v.lora` | Self-attention Q/K/V projections |
| `cross_attn` | `attn.add_q/k/v_proj`, `attn.to_add_out`, `attn2.to_*`, `ff_context.*` | Cross-attention with text stream |
| `ffn` | `ff.linear_in/out`, `ff.net.*`, `ffn.net.*`, `mlp.*`, `feed_forward.*` | Feed-forward network |
| `norm` | `norm` | Layer/group normalization |

Parameters that don't match any pattern get `default_lr`.

### Which models have which components

Not all models have all component types:

| Model Family | self_attn | cross_attn | ffn | Notes |
|-------------|-----------|------------|-----|-------|
| **Flux 2 / Klein** | `attn.to_q/k/v/out` | `attn.add_q/k/v_proj`, `ff_context.*` | `ff.linear_in/out` | Double-stream: self_attn = image, cross_attn = text |
| **SD 1.5 / SDXL** | `attn1.to_q/k/v` | `attn2.to_q/k/v` | `ff.net.*` | Classic U-Net cross-attention |
| **SD3 / SD3.5** | `attn.to_q/k/v` | -- | `ff.net.*` | Joint attention (no separate cross) |
| **WAN 2.x** | `attn1.to_q/k/v` | `attn2.to_q/k/v` | `ffn.net.*` | Video model with self + cross |
| **LTX-2** | `to_q/k/v` (bare) | -- | `ff.net.*` | Bare names, no `attn.` prefix |
| **Qwen** | `attn.to_q/k/v` | -- | `mlp.gate/up/down_proj` | Gated MLP |
| **ZImage** | `attention.to_q/k/v` | -- | `feed_forward.*` | Uses `attention.` prefix |
| **Chroma / HunyuanVideo / PixArt / Sana** | `attn.to_q/k/v` | -- | `ff.net.*` | Standard naming |

## Model-Specific Examples

### Flux 2 Klein 4B -- Style LoRA

Small model, fast iteration. Emphasize self-attention for style capture, moderate FFN, low cross-attention to preserve prompt following.

```json
{
  "model_type": "flux_2_klein_4b",
  "learning_rate": 0.0004,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0005,
      "cross_attn": 0.00015,
      "ffn": 0.0003,
      "norm": 0.00005
    },
    "depth_strategy": "inverted_u",
    "min_depth_factor": 0.4,
    "peak_position": 0.5,
    "default_lr": 0.0003
  }
}
```

**Why inverted_u**: Klein 4B has fewer blocks. Mid-level blocks capture the most style information. Peak at 0.5 gives highest LR to the middle of the network.

### Flux 2 Klein 9B -- Subject LoRA

Larger model, subject fidelity matters. Higher self-attention LR, preserve text alignment with low cross-attention LR.

```json
{
  "model_type": "flux_2_klein_9b",
  "learning_rate": 0.0004,
  "adapter": {
    "type": "lora",
    "rank": 32,
    "alpha": 32
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "cross_attn": 0.0001,
      "ffn": 0.00025,
      "norm": 0.00003
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.3,
    "default_lr": 0.00025
  }
}
```

**Why linear**: For subject LoRA, shallow blocks learn identity features (face structure, body shape). Deep blocks are more semantic and should change less.

### Flux 2 Dev -- Full Finetune

When training all parameters (no LoRA), per-layer LR prevents catastrophic forgetting in deep blocks.

```json
{
  "model_type": "flux_2_dev",
  "training_method": "fine_tune",
  "learning_rate": 0.00001,
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.000015,
      "cross_attn": 0.000005,
      "ffn": 0.00001,
      "norm": 0.000002
    },
    "depth_strategy": "cosine",
    "min_depth_factor": 0.2,
    "default_lr": 0.00001
  }
}
```

**Why cosine**: Smoother decay prevents abrupt LR changes between adjacent blocks. The 0.2 min factor means deep blocks still learn at 1/5 the rate of shallow blocks.

### SD 1.5 -- Character LoRA

Classic model with separate self-attention and cross-attention blocks.

```json
{
  "model_type": "sd15",
  "learning_rate": 0.0001,
  "adapter": {
    "type": "lora",
    "rank": 8,
    "alpha": 8
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.00015,
      "cross_attn": 0.00005,
      "ffn": 0.0001,
      "norm": 0.00002
    },
    "depth_strategy": "inverted_u",
    "min_depth_factor": 0.3,
    "peak_position": 0.4,
    "default_lr": 0.0001
  }
}
```

**Why peak_position=0.4**: U-Net architecture has down/mid/up blocks. The most expressive features are in the early-mid range. Shifting peak earlier captures this.

### SDXL -- Concept LoRA

```json
{
  "model_type": "sdxl",
  "learning_rate": 0.0001,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.00012,
      "cross_attn": 0.00004,
      "ffn": 0.00008,
      "norm": 0.00001
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.35,
    "default_lr": 0.00008
  }
}
```

### SD3.5 -- Style Transfer

SD3 uses joint attention (no separate cross-attention). All text/image interaction happens through self-attention.

```json
{
  "model_type": "sd35",
  "learning_rate": 0.0002,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.00025,
      "ffn": 0.00015,
      "norm": 0.00003
    },
    "depth_strategy": "cosine",
    "min_depth_factor": 0.3,
    "default_lr": 0.00015
  }
}
```

**Note**: No `cross_attn` entry needed -- SD3 has no separate cross-attention.

### LTX-2 -- Video LoRA with Stagehand

LTX-2 has 48 transformer blocks. With Stagehand streaming blocks through GPU one at a time, per-layer LR adds zero VRAM overhead (it only affects optimizer param group construction).

```json
{
  "model_type": "ltx2",
  "learning_rate": 0.0003,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "ffn": 0.00025,
      "norm": 0.00003
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.2,
    "default_lr": 0.00025
  },
  "memory": {
    "stagehand": {
      "enabled": true,
      "pinned_pool_mb": 8192,
      "prefetch_window": 2
    }
  }
}
```

**Why min_depth_factor=0.2**: With 48 blocks, a 0.3 min factor means block 47 gets lr * 0.3. With 0.2, the deepest blocks change even less -- good for video temporal consistency.

### WAN 2.1 -- Video LoRA (Dual-Stage)

WAN has both self-attention (attn1) and cross-attention (attn2). The dual-stage path (WAN 2.2) also supports per-layer LR.

```json
{
  "model_type": "wan",
  "learning_rate": 0.0003,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "cross_attn": 0.00015,
      "ffn": 0.00025,
      "norm": 0.00003
    },
    "depth_strategy": "cosine",
    "min_depth_factor": 0.25,
    "default_lr": 0.00025
  }
}
```

### Qwen -- Image Generation LoRA

Qwen uses gated MLP (gate_proj, up_proj, down_proj) instead of standard FFN.

```json
{
  "model_type": "qwen",
  "learning_rate": 0.0002,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.00025,
      "ffn": 0.00015,
      "norm": 0.00002
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.3,
    "default_lr": 0.00015
  }
}
```

### ZImage -- Turbo LoRA

ZImage uses `attention.to_q/k/v` and `feed_forward.*` naming.

```json
{
  "model_type": "zimage",
  "learning_rate": 0.0003,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "ffn": 0.00025,
      "norm": 0.00003
    },
    "depth_strategy": "inverted_u",
    "min_depth_factor": 0.35,
    "peak_position": 0.45,
    "default_lr": 0.00025
  }
}
```

### Chroma -- Artistic LoRA

```json
{
  "model_type": "chroma_1",
  "learning_rate": 0.0003,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.0004,
      "ffn": 0.0002,
      "norm": 0.00003
    },
    "depth_strategy": "cosine",
    "min_depth_factor": 0.3,
    "default_lr": 0.0002
  }
}
```

### HunyuanVideo -- Video LoRA

```json
{
  "model_type": "hunyuan_video",
  "learning_rate": 0.0003,
  "adapter": {
    "type": "lora",
    "rank": 16,
    "alpha": 16
  },
  "per_layer_lr": {
    "enabled": true,
    "component_lr": {
      "self_attn": 0.00035,
      "ffn": 0.0002,
      "norm": 0.00003
    },
    "depth_strategy": "linear",
    "min_depth_factor": 0.25,
    "default_lr": 0.0002
  }
}
```

## Strategy Guide by Use Case

### Style LoRA (artistic style transfer)

Mid-level features matter most. Use `inverted_u` with peak at 0.4-0.6.

```json
"depth_strategy": "inverted_u",
"min_depth_factor": 0.3,
"peak_position": 0.5,
"component_lr": {
  "self_attn": 0.0005,
  "cross_attn": 0.0002,
  "ffn": 0.0003
}
```

### Subject LoRA (person, character, object)

Shallow blocks learn identity. Deep blocks should change less. Use `linear`.

```json
"depth_strategy": "linear",
"min_depth_factor": 0.3,
"component_lr": {
  "self_attn": 0.0004,
  "cross_attn": 0.0001,
  "ffn": 0.00025
}
```

### Concept LoRA (new concept, token binding)

Cross-attention is where text-to-image binding lives. Give it moderate LR. Use `cosine` for smooth decay.

```json
"depth_strategy": "cosine",
"min_depth_factor": 0.35,
"component_lr": {
  "self_attn": 0.0003,
  "cross_attn": 0.00025,
  "ffn": 0.0002
}
```

### Full Finetune (catastrophic forgetting prevention)

Much lower LRs. Cosine decay with aggressive min factor to protect deep semantic blocks.

```json
"depth_strategy": "cosine",
"min_depth_factor": 0.15,
"component_lr": {
  "self_attn": 0.000015,
  "cross_attn": 0.000005,
  "ffn": 0.00001
}
```

### Component-Only (no depth variation)

When you only want different LR per component, not per depth:

```json
"depth_strategy": "flat",
"component_lr": {
  "self_attn": 0.0004,
  "cross_attn": 0.0001,
  "ffn": 0.0003
}
```

### Conservative (minimal intervention)

Small depth variation, mostly flat. Good starting point if unsure.

```json
"depth_strategy": "linear",
"min_depth_factor": 0.7,
"component_lr": {
  "self_attn": 0.0004,
  "cross_attn": 0.0003,
  "ffn": 0.00035
}
```

## Startup Log Output

When enabled, the full 2D LR grid is printed at startup:

```
[per_layer_lr] model_type=flux_2_klein_4b strategy=linear min_depth_factor=0.3 groups=24 total_params=192
  transformer_blocks.000.cross_attn                   lr=0.000200  params=8
  transformer_blocks.000.ffn                          lr=0.000300  params=4
  transformer_blocks.000.self_attn                    lr=0.000400  params=8
  transformer_blocks.001.cross_attn                   lr=0.000177  params=8
  transformer_blocks.001.ffn                          lr=0.000267  params=4
  transformer_blocks.001.self_attn                    lr=0.000356  params=8
  ...
  transformer_blocks.011.cross_attn                   lr=0.000060  params=8
  transformer_blocks.011.ffn                          lr=0.000090  params=4
  transformer_blocks.011.self_attn                    lr=0.000120  params=8
  single_transformer_blocks.000.ffn                   lr=0.000300  params=4
  single_transformer_blocks.000.self_attn             lr=0.000400  params=8
  ...
```

This makes it easy to verify the schedule is what you intended before training starts.

## Interaction with Stagehand

Per-layer LR is orthogonal to Stagehand. It only affects optimizer param group construction at startup. During training:

- Stagehand streams blocks through GPU as usual
- The optimizer already has per-group LRs assigned
- No additional VRAM overhead
- No additional compute overhead
- Works with both block mode and layer mode Stagehand

The LR scheduler (cosine, linear, constant, etc.) applies its multiplicative factor to each param group's LR independently, so warmup and decay work correctly with per-layer LR.

## Interaction with LR Schedulers

PyTorch's `LambdaLR` scheduler multiplies each param group's base LR by a factor. With per-layer LR, each group has a different base LR, so:

- Warmup ramps all groups proportionally
- Cosine/linear decay scales all groups proportionally
- The relative ratios between groups are preserved throughout training

For example, if self_attn starts at 0.0004 and ffn starts at 0.0003, after a cosine decay to 50%:
- self_attn = 0.0002
- ffn = 0.00015

The 4:3 ratio is maintained.
