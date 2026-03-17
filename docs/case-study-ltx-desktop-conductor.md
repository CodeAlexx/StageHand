# Case Study: Porting the Conductor to LTX-Desktop (LTX-2 19B LoRA on 24 GB)

> **Note**: This case study covers porting Stagehand's `LayerOffloadConductor` pattern into [LTX Desktop](https://github.com/Lightricks/LTX-Desktop), an Electron app for LTX-2 video generation. The upstream trainer is [ltx-trainer](https://github.com/Lightricks/LTX-2) from Lightricks.

Training LTX-2's 18.88B-parameter audio-video transformer with LoRA on a single 24 GB GPU. 90% of 48 blocks offloaded to CPU. 8.17 GB VRAM.

The port surfaced a critical incompatibility between `use_reentrant=True` gradient checkpointing and models that pass dataclass objects (not raw tensors) between blocks.

## The Problem

LTX-2 is Lightricks' audio-video generation model:

| Component | Parameters | bf16 Size |
|-----------|-----------|-----------|
| Transformer (LTXModel, 48 blocks) | ~18.88B | ~37.8 GB |
| Text encoder (Gemma 3 12B) | ~12B | ~24 GB |
| Video VAE | ~200M | ~400 MB |
| **Total** | **~31B** | **~62 GB** |

The transformer alone exceeds 24 GB. The ltx-trainer's built-in gradient checkpointing saves activation memory but doesn't address weight residency. Quantization (INT8) was tried but `QBytesTensor` objects don't serialize correctly with block swapping. The conductor pattern — proven on Flux at 4.15 GB — was the right tool.

## The Port

The conductor was adapted from Serenity's `serenity/memory/conductor/offload.py` and `serenity/memory/checkpoint_layer.py` into LTX-Desktop's backend:

```
backend/services/training/
    conductor.py              # LayerOffloadConductor
    checkpoint_layer.py       # OffloadCheckpointLayer
    stagehand_trainer.py      # Monkey-patches LtxvTrainer
    config_builder.py         # TrainingRequest → LtxTrainerConfig
```

`StagehandTrainer` monkey-patches two methods on the upstream `LtxvTrainer`:

1. **`_prepare_models_for_training`**: After LoRA is applied, wraps all 48 blocks with `OffloadCheckpointLayer`, creates a conductor with `layer_offload_fraction=0.9` (5 blocks on GPU, 43 offloaded), disables the model's built-in gradient checkpointing (OffloadCheckpointLayer handles it), and calls `accelerator.prepare(device_placement=False)`.

2. **`_training_step`**: Manages conductor lifecycle (`end_step` / `start_forward`), replicates the upstream embedding processing, calls the transformer forward, returns loss.

### Why Conductor, Not StagehandRuntime

StagehandRuntime is designed for **inference**. It installs forward hooks inside a `managed_forward()` context manager. During training, gradient checkpointing replays the forward pass during backward — but by then the context has exited and the hooks are gone. Blocks can't be loaded back for recomputation.

The conductor pattern keeps offload management **permanently active** via `OffloadCheckpointLayer` wrappers around each block, not via temporary hooks.

## The Bug: Gradient Disconnection

After the port, the first training step failed:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The error occurs during `accelerator.backward(loss)`. The loss tensor has no gradient graph.

### Root Cause: Dataclass Block Interface

Serenity's Flux blocks accept and return **raw tensors**:

```python
# Flux block interface (works with use_reentrant=True)
def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, ...) -> tuple[Tensor, Tensor]:
```

LTX-2's blocks accept and return **frozen dataclass objects**:

```python
@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor              # hidden states (modified per block)
    context: torch.Tensor        # text embeddings
    timesteps: torch.Tensor      # adaln timestep embeddings
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    # ... 6 more tensor fields
    enabled: bool

# LTX block interface (breaks with use_reentrant=True)
def forward(self, video: TransformerArgs, audio: TransformerArgs, perturbations) -> tuple[TransformerArgs, TransformerArgs]:
```

With `use_reentrant=True`, `torch.utils.checkpoint.checkpoint()` uses `torch.autograd.Function.apply()` internally. This mechanism attaches `grad_fn` only to **top-level tensor outputs**. When the block returns `(TransformerArgs, TransformerArgs)`, the autograd machinery sees non-tensor objects — it can't attach `grad_fn` to them. The `.x` tensors nested inside were produced under `no_grad` (the reentrant checkpoint's forward context) and remain detached.

The dummy tensor trick (passing a `requires_grad=True` zero tensor as checkpoint input) doesn't help — it ensures checkpoint *runs* backward, but the output tensors still have no `grad_fn` because the autograd Function only wraps top-level tensor returns.

The ltx-trainer itself uses `use_reentrant=False` in its own gradient checkpointing for exactly this reason:

```python
# From ltx_core/model/transformer/model.py
video, audio = torch.utils.checkpoint.checkpoint(
    block, video, audio, perturbations,
    use_reentrant=False,  # ← Required for dataclass args/returns
)
```

### The Fix

Two changes:

**1. `checkpoint_layer.py`**: Always use `use_reentrant=False`.

With `use_reentrant=False`, checkpoint uses **saved-tensor hooks** instead of `torch.autograd.Function`. The forward runs with `enable_grad` normally — tensors inside dataclass returns get proper `grad_fn` from the standard autograd graph. Checkpoint replaces memory storage with recomputation transparently.

```python
# Before (broken with dataclass block interface)
output = torch.utils.checkpoint.checkpoint(
    _checkpointed_forward,
    self._dummy,       # dummy tensor trick
    *args, *kw_vals,
    use_reentrant=True,
)

# After (works with any block interface)
output = torch.utils.checkpoint.checkpoint(
    _checkpointed_forward,
    *args, *kw_vals,
    use_reentrant=False,
)
```

The dummy tensor and reentrant code path are removed entirely.

**2. `conductor.py`**: Replace `torch.is_grad_enabled()` detection with per-block call counts.

The conductor must distinguish forward from backward recompute:
- **Forward**: load block → compute → evict to CPU
- **Backward recompute**: load block → compute → keep on GPU (gradients need it)

With `use_reentrant=True`, forward runs under `no_grad` and backward under `enable_grad`, so `torch.is_grad_enabled()` is a reliable signal. With `use_reentrant=False`, **both** run under `enable_grad` — the signal is gone.

The fix uses per-block call counts. Each block is called exactly once during forward and once during backward recompute (checkpoint replays it). First call = forward, second call = backward:

```python
def start_forward(self) -> None:
    self._block_call_counts = [0] * len(self._layers)
    # ...

def before_layer(self, layer_index, ...):
    self._block_call_counts[layer_index] += 1
    if self._block_call_counts[layer_index] > 1 and self._is_forward_pass:
        self._is_forward_pass = False  # Transition to backward mode
    # ...
```

## Hardware

| Resource | Used |
|----------|------|
| GPU | RTX 3090 (24 GB VRAM) |
| VRAM allocated | 8.17 GB |
| System RAM | 62 GB |
| Blocks on GPU | 5 of 48 |
| Blocks on CPU | 43 of 48 |

## Training Configuration

```python
quantization       = None          # Conductor handles VRAM
optimizer_type     = "adamw8bit"   # 8-bit optimizer saves ~75% optimizer memory
gradient_checkpointing = True      # Via OffloadCheckpointLayer, NOT model-level
layer_offload_fraction = 0.9       # Keep ~5 of 48 blocks on GPU
offload_activations    = True      # Move activations to CPU between blocks
enable_async           = True      # Non-blocking CUDA transfers
mixed_precision_mode   = "bf16"
lora_rank              = 16
lora_targets           = ["to_k", "to_q", "to_v", "to_out.0"]
```

## Key Lesson

**`use_reentrant=True` only works when blocks accept and return raw tensors.**

If the model uses dataclasses, named tuples, or any non-tensor wrapper around hidden states, you **must** use `use_reentrant=False` and replace `torch.is_grad_enabled()` forward/backward detection with an alternative signal (call counts, explicit flags, etc.).

Before porting the conductor to a new model, always check the block's type signature:

```python
# Safe for use_reentrant=True
def forward(self, x: Tensor, ...) -> Tensor: ...
def forward(self, x: Tensor, ...) -> tuple[Tensor, Tensor]: ...

# Requires use_reentrant=False
def forward(self, video: TransformerArgs, ...) -> tuple[TransformerArgs, ...]: ...
def forward(self, x: SomeDataclass, ...) -> SomeDataclass: ...
```

## Files

| File | Repository |
|------|------------|
| `conductor.py` (adapted) | LTX-Desktop `backend/services/training/` |
| `checkpoint_layer.py` (adapted) | LTX-Desktop `backend/services/training/` |
| `stagehand_trainer.py` | LTX-Desktop `backend/services/training/` |
| `LayerOffloadConductor` (original) | Serenity `serenity/memory/conductor/offload.py` |
| `OffloadCheckpointLayer` (original) | Serenity `serenity/memory/checkpoint_layer.py` |
| Gradient flow tests | Serenity `serenity/tests/test_offload_checkpoint_gradient_flow.py` |
| `BasicAVTransformerBlock` | ltx_core `model/transformer/transformer.py` |
| `TransformerArgs` | ltx_core `model/transformer/transformer_args.py` |
