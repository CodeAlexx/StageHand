# Residency Protection

Control which blocks stay on GPU and prevent eviction during multi-model workflows.

## Problem

When multiple models share VRAM through Stagehand, the scheduler evicts blocks based on distance-to-next-use scoring. This works well for single-model streaming, but breaks down when:

- A model must stay fully loaded across multiple inference calls (e.g. text encoder used repeatedly)
- A primary model must not lose blocks when a guest model (upscaler, VAE) is temporarily loaded
- You need explicit VRAM budgeting across multiple runtimes

## API

### `keep_resident()` — temporary eviction suppression

Context manager that prevents blocks from being selected as eviction candidates.

```python
with runtime.keep_resident():
    # All blocks of this runtime are protected from eviction
    result1 = model.generate(prompt1)
    result2 = model.generate(prompt2)
# Protection lifted — blocks are normal eviction candidates again
```

Protect another runtime's blocks:

```python
with transformer_runtime.keep_resident(text_encoder_runtime):
    # Text encoder blocks stay on GPU while transformer runs
    embeddings = text_encoder(prompt)
    output = transformer(embeddings)
```

**Behavior:**
- Sets protection flag on all blocks in the target runtime's registry
- Protected blocks are skipped during eviction candidate selection
- Protection is automatically removed on context exit (even on exception)
- If no blocks are found, logs a warning and yields without effect

### `reserve_for_resident()` — persistent VRAM reservation

Marks a model as a long-term resident and subtracts its VRAM footprint from the available guest budget.

```python
# Reserve 2GB for the text encoder — scheduler accounts for this
runtime.reserve_for_resident(text_encoder_runtime, priority=ResidentPriority.PRIMARY)

# ... run many steps, text encoder blocks never evicted ...

# Release when done with the text encoder
runtime.release_reservation(text_encoder_runtime)
```

**Parameters:**
- `model_or_runtime`: Target runtime to reserve (or `None` for self)
- `priority`: `ResidentPriority.PRIMARY` (default, never evicted) or `ResidentPriority.SECONDARY` (evicted only under extreme pressure)

**Effects:**
- `PRIMARY` blocks are added to the scheduler's protection set
- `BudgetManager` subtracts the model's total parameter size from available guest headroom
- Guest models loaded via `as_guest()` are evaluated against remaining headroom only

### `release_reservation()` — free reserved VRAM

```python
runtime.release_reservation(text_encoder_runtime)
```

Removes the reservation, unprotects blocks (if PRIMARY), and returns headroom to the guest pool.

### `as_guest()` — scoped guest model loading

Context manager for short-lived models that should only use unreserved VRAM.

```python
runtime.reserve_for_resident(priority=ResidentPriority.PRIMARY)

with runtime.as_guest(upscaler_runtime):
    # Upscaler runs within guest headroom only
    # Primary runtime blocks are never evicted to make room
    result = upscaler.run(latent)
# Guest evicted on context exit
```

## Priority Levels

| Priority | Eviction behavior | Use case |
|----------|------------------|----------|
| `PRIMARY` | Never evicted while reservation active | Main transformer, text encoder |
| `SECONDARY` | Evicted only under extreme VRAM pressure | Auxiliary models, optional components |

## Example: LTX-2 Four-Pass Pipeline

The four-pass video generation pipeline uses residency protection to manage a 22B transformer, text encoder, spatial upscaler, and temporal upscaler on a single 24GB GPU:

```python
# Stage 1: Text encoding
with transformer_runtime.keep_resident(text_encoder_runtime):
    embeddings = text_encoder(prompt)

# Stage 2: Base generation (transformer streams through VRAM)
transformer_runtime.reserve_for_resident(priority=ResidentPriority.PRIMARY)
with transformer_runtime.managed_forward():
    base_latent = denoise(transformer, noise, embeddings, steps=8)

# Stage 3: Spatial upscale (guest model, uses remaining headroom)
with transformer_runtime.as_guest(spatial_upscaler_runtime):
    upscaled = spatial_upscaler(base_latent)

# Stage 4: Temporal upscale (another guest)
with transformer_runtime.as_guest(temporal_upscaler_runtime):
    final = temporal_upscaler(upscaled)

transformer_runtime.release_reservation()
```

## Internals

- `keep_resident()` calls `scheduler.protect_blocks(block_ids)` / `unprotect_blocks(block_ids)`
- `reserve_for_resident()` stores a `_ReservationEntry` dataclass (runtime ref, size, block IDs, priority)
- `BudgetManager.reserve_bytes()` / `release_reserved_bytes()` adjust the effective watermarks
- Reservations are tracked by `id(target)` — one reservation per runtime instance
- `release_reservation()` is idempotent (no-op if no reservation exists)
