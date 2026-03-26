import os
import pytest

if not os.environ.get("VMM_GPU_TESTS"):
    pytest.skip("Set VMM_GPU_TESTS=1 to run GPU tests", allow_module_level=True)

import torch
import stagehand_vmm as vmm


def test_allocator_creation():
    """Create allocator, verify stats."""
    alloc = vmm.SlabAllocator(device=0)
    s = alloc.stats()
    assert s["granularity"] > 0
    assert s["vram_ceiling"] > 0


def test_slab_lifecycle():
    """Create slab, define regions, destroy."""
    alloc = vmm.SlabAllocator(device=0)
    slab = alloc.create_slab(1024 * 1024 * 64)
    r0 = alloc.define_region(slab, 0, 1024 * 1024 * 32)
    r1 = alloc.define_region(slab, 1024 * 1024 * 32, 1024 * 1024 * 32)
    alloc.destroy_slab(slab)


def test_ensure_resident_and_tensor():
    """Map a region, get a tensor, verify it's on CUDA."""
    alloc = vmm.SlabAllocator(device=0)
    slab = alloc.create_slab(1024 * 1024 * 64)
    region = alloc.define_region(slab, 0, 1024 * 1024 * 32)
    alloc.set_priority(slab, 100)

    stream = torch.cuda.current_stream().cuda_stream
    handle = alloc.ensure_resident(slab, region, stream)

    # 32MB / 2 bytes per bf16 = 16M elements
    tensor = handle.as_tensor("bfloat16", [16 * 1024 * 1024])
    assert tensor.device.type == "cuda"
    assert tensor.dtype == torch.bfloat16
    assert tensor.shape[0] == 16 * 1024 * 1024

    del handle
    alloc.destroy_slab(slab)


def test_handle_release():
    """Explicit release, then as_tensor should fail."""
    alloc = vmm.SlabAllocator(device=0)
    slab = alloc.create_slab(1024 * 1024 * 64)
    region = alloc.define_region(slab, 0, 1024 * 1024 * 32)
    alloc.set_priority(slab, 100)

    stream = torch.cuda.current_stream().cuda_stream
    handle = alloc.ensure_resident(slab, region, stream)
    handle.release()

    with pytest.raises(RuntimeError):
        handle.as_tensor("float32", [1024])

    alloc.destroy_slab(slab)


def test_watermark_raises():
    """Force watermark, verify MemoryError."""
    alloc = vmm.SlabAllocator(device=0, ceiling_mb=64)

    slab1 = alloc.create_slab(1024 * 1024 * 48)
    r1 = alloc.define_region(slab1, 0, 1024 * 1024 * 48)
    alloc.set_priority(slab1, 50)

    slab2 = alloc.create_slab(1024 * 1024 * 48)
    r2 = alloc.define_region(slab2, 0, 1024 * 1024 * 48)
    alloc.set_priority(slab2, 100)

    stream = torch.cuda.current_stream().cuda_stream

    h1 = alloc.ensure_resident(slab1, r1, stream)
    del h1  # release so eviction can proceed

    h2 = alloc.ensure_resident(slab2, r2, stream)

    # slab1 should now be watermarked
    with pytest.raises(MemoryError):
        alloc.ensure_resident(slab1, r1, stream)

    del h2
    alloc.destroy_slab(slab1)
    alloc.destroy_slab(slab2)


def test_tensor_survives_handle_release():
    """DLPack tensor stays valid after handle.release()."""
    alloc = vmm.SlabAllocator(device=0)
    slab = alloc.create_slab(1024 * 1024 * 64)
    region = alloc.define_region(slab, 0, 1024 * 1024 * 32)
    alloc.set_priority(slab, 100)

    stream = torch.cuda.current_stream().cuda_stream
    handle = alloc.ensure_resident(slab, region, stream)
    tensor = handle.as_tensor("float32", [8 * 1024 * 1024])

    handle.release()

    # Tensor should still work — DLPack context holds its own refcount
    result = tensor + 1
    assert result.shape == tensor.shape

    del tensor
    alloc.destroy_slab(slab)


def test_matmul_on_vmm_tensor():
    """Verify VMM-backed tensors work in compute."""
    alloc = vmm.SlabAllocator(device=0)
    slab = alloc.create_slab(1024 * 1024 * 8)
    region = alloc.define_region(slab, 0, 1024 * 1024 * 4)
    alloc.set_priority(slab, 100)

    stream = torch.cuda.current_stream().cuda_stream
    handle = alloc.ensure_resident(slab, region, stream)

    # 4MB / 4 bytes = 1M elements = 1024x256 float32
    a = handle.as_tensor("float32", [1024, 256])
    b = torch.randn(256, 512, device="cuda")

    c = torch.matmul(a, b)
    assert c.shape == (1024, 512)

    del handle
    alloc.destroy_slab(slab)
