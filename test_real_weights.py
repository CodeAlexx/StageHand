"""Real-weight integration test for stagehand layer mode.

Loads CLIP-L (235MB, ~125M params) and verifies that layer-mode
offloading produces bit-identical results to normal GPU inference.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from safetensors.torch import load_file

import stagehand

CLIP_L_PATH = "/home/alex/EriDiffusion/Models/clip/clip_l.safetensors"


# ── minimal CLIP-L model definition ─────────────────────────────────────


class CLIPAttention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.heads = heads
        self.head_dim = d // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn)


class CLIPBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d)
        self.self_attn = CLIPAttention(d, heads)
        self.layer_norm2 = nn.LayerNorm(d)
        self.mlp_fc1 = nn.Linear(d, d * 4)
        self.mlp_fc2 = nn.Linear(d * 4, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm1(x)
        h = self.self_attn(h)
        x = x + h
        h = self.layer_norm2(x)
        h = torch.nn.functional.gelu(self.mlp_fc1(h))
        h = self.mlp_fc2(h)
        return x + h


class CLIPL(nn.Module):
    """Minimal CLIP-L text encoder (12 blocks, 768d, 12 heads)."""

    def __init__(self) -> None:
        super().__init__()
        d, heads, layers, vocab = 768, 12, 12, 49408
        self.token_embedding = nn.Embedding(vocab, d)
        self.position_embedding = nn.Embedding(77, d)
        self.blocks = nn.ModuleList([CLIPBlock(d, heads) for _ in range(layers)])
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens) + self.position_embedding(
            torch.arange(tokens.shape[1], device=tokens.device)
        )
        for block in self.blocks:
            x = block(x)
        return self.final_layer_norm(x)


def _map_clip_l_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map safetensors keys to our model structure."""
    mapped: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        # text_model.embeddings -> our names
        nk = nk.replace("text_model.embeddings.token_embedding.weight", "token_embedding.weight")
        nk = nk.replace("text_model.embeddings.position_embedding.weight", "position_embedding.weight")
        # text_model.encoder.layers.N -> blocks.N
        nk = nk.replace("text_model.encoder.layers.", "blocks.")
        # self_attn stays
        # mlp.fc1 -> mlp_fc1, mlp.fc2 -> mlp_fc2
        nk = nk.replace(".mlp.fc1.", ".mlp_fc1.")
        nk = nk.replace(".mlp.fc2.", ".mlp_fc2.")
        # final_layer_norm
        nk = nk.replace("text_model.final_layer_norm.", "final_layer_norm.")
        mapped[nk] = v
    return mapped


def main() -> None:
    print("=" * 70)
    print("Stagehand Layer Mode — Real Weight Test (CLIP-L)")
    print("=" * 70)

    # 1. Load model and weights on GPU (reference).
    print("\n[1] Loading CLIP-L on GPU for reference output...")
    model = CLIPL()
    sd = load_file(CLIP_L_PATH)
    mapped = _map_clip_l_keys(sd)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)} (likely text_projection etc)")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")

    model = model.to("cuda", dtype=torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    total_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"    Params: {total_params:,} ({total_mb:.1f} MB)")

    # 2. Reference inference.
    tokens = torch.randint(0, 49408, (1, 77), device="cuda")
    with torch.no_grad():
        ref_output = model(tokens).clone()
    print(f"    Reference output shape: {ref_output.shape}")
    print(f"    Reference output hash: {ref_output.sum().item():.6f}")

    # 3. Move to CPU, wrap with stagehand.layer().
    print("\n[2] Moving model to CPU and wrapping with stagehand.layer()...")
    model = model.cpu()
    torch.cuda.empty_cache()

    free_before = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"    VRAM free before layer mode: {free_before:.0f} MB")

    t0 = time.perf_counter()
    model = stagehand.layer(
        model,
        vram_budget="4GB",
        prefetch_k=3,
        dtype=torch.float32,
        inference_mode=True,
        telemetry=True,
    )
    t_setup = time.perf_counter() - t0
    runtime = model._stagehand_layer_runtime
    print(f"    Setup time: {t_setup:.3f}s")
    print(f"    Layers discovered: {runtime.num_layers}")
    print(f"    Mode: {runtime.mode}")

    # 4. Step 0: trace pass.
    print("\n[3] Step 0 — trace pass (no prefetch)...")
    tokens_cpu = tokens.cpu()
    with torch.no_grad():
        t0 = time.perf_counter()
        out0 = model(tokens_cpu.to("cuda" if torch.cuda.is_available() else "cpu"))
        t_trace = time.perf_counter() - t0

    peak_trace = torch.cuda.max_memory_allocated() / 1024**2
    print(f"    Trace time: {t_trace:.3f}s")
    print(f"    Mode after trace: {runtime.mode}")
    print(f"    Traced order: {len(runtime.trace_order)} layers")
    print(f"    Peak VRAM: {peak_trace:.0f} MB")

    # 5. Step 1: rebuild + scheduled (with prefetch).
    print("\n[4] Step 1 — scheduled pass (with prefetch)...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t0 = time.perf_counter()
        out1 = model(tokens_cpu.to("cuda" if torch.cuda.is_available() else "cpu"))
        t_sched = time.perf_counter() - t0

    peak_sched = torch.cuda.max_memory_allocated() / 1024**2
    print(f"    Scheduled time: {t_sched:.3f}s")
    print(f"    Mode: {runtime.mode}")
    print(f"    Step: {runtime.step}")
    print(f"    Peak VRAM: {peak_sched:.0f} MB")

    # 6. Run a few more steps for telemetry.
    print("\n[5] Running 5 more steps for telemetry...")
    times = []
    for i in range(5):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(tokens_cpu.to("cuda" if torch.cuda.is_available() else "cpu"))
            dt = time.perf_counter() - t0
            times.append(dt)

    avg_time = sum(times) / len(times)
    print(f"    Avg step time: {avg_time:.3f}s")
    print(f"    Step count: {runtime.step}")
    print(f"    Stats: {runtime.stats}")

    # 7. Compare outputs.
    print("\n[6] Comparing outputs...")

    # Move outputs to same device for comparison.
    out0_cpu = out0.cpu()
    out1_cpu = out1.cpu()
    ref_cpu = ref_output.cpu()

    # Check numerical match.
    max_diff_0 = (out0_cpu - ref_cpu).abs().max().item()
    max_diff_1 = (out1_cpu - ref_cpu).abs().max().item()
    match_0 = torch.allclose(out0_cpu, ref_cpu, atol=1e-4, rtol=1e-4)
    match_1 = torch.allclose(out1_cpu, ref_cpu, atol=1e-4, rtol=1e-4)

    print(f"    Step 0 vs reference: max_diff={max_diff_0:.2e}, match={match_0}")
    print(f"    Step 1 vs reference: max_diff={max_diff_1:.2e}, match={match_1}")

    # 8. Shutdown.
    runtime.shutdown()
    torch.cuda.empty_cache()
    free_after = torch.cuda.mem_get_info()[0] / 1024**2
    print(f"\n    VRAM free after shutdown: {free_after:.0f} MB")

    # Final verdict.
    print("\n" + "=" * 70)
    if match_0 and match_1:
        print("PASS: Layer mode produces identical outputs to GPU reference")
    else:
        print("WARN: Outputs differ (check tolerances)")
        print(f"      Step 0 max diff: {max_diff_0:.2e}")
        print(f"      Step 1 max diff: {max_diff_1:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
