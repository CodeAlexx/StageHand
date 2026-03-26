//! Integration tests for stagehand-vmm.
//!
//! These tests require an NVIDIA GPU and are gated behind the `gpu-tests` feature.
//! Run with: `cargo test --features gpu-tests`

#![cfg(feature = "gpu-tests")]

use stagehand_vmm::{SlabAllocator, VmmError};

fn init_cuda() {
    #[link(name = "cuda")]
    extern "C" {
        fn cuInit(flags: u32) -> u32;
    }
    // SAFETY: cuInit(0) initializes the CUDA driver. Safe to call multiple times.
    unsafe {
        let result = cuInit(0);
        assert_eq!(result, 0, "cuInit failed with error {result}");
    }
}

fn create_allocator() -> SlabAllocator {
    init_cuda();
    SlabAllocator::new(0, None).expect("failed to create allocator")
}

#[test]
fn test_create_destroy_slab() {
    let alloc = create_allocator();
    let slab = alloc.create_slab(64 * 1024 * 1024).expect("create_slab failed");
    alloc.destroy_slab(slab).expect("destroy_slab failed");
}

#[test]
fn test_define_regions() {
    let alloc = create_allocator();
    let gran = alloc.granularity();
    let slab = alloc.create_slab(gran * 10).expect("create_slab failed");

    let mut ids = Vec::new();
    for i in 0..10 {
        let id = alloc
            .define_region(slab, i * gran, gran)
            .expect("define_region failed");
        ids.push(id);
    }

    assert_eq!(ids.len(), 10);
    for (expected, actual) in ids.iter().enumerate() {
        assert_eq!(*actual, expected);
    }

    alloc.destroy_slab(slab).expect("destroy_slab failed");
}

#[test]
fn test_ensure_resident_maps_region() {
    let alloc = create_allocator();
    let gran = alloc.granularity();
    let slab = alloc.create_slab(gran).expect("create_slab failed");
    let region = alloc.define_region(slab, 0, gran).expect("define_region failed");

    let stream = std::ptr::null_mut();
    let handle = alloc
        .ensure_resident(slab, region, stream)
        .expect("ensure_resident failed");

    // SAFETY: allocator is alive and region is mapped.
    assert_ne!(unsafe { handle.as_ptr() }, 0);

    let stats = alloc.stats();
    assert_eq!(stats.total_slabs, 1);
    assert_eq!(stats.total_regions, 1);
    assert!(stats.mapped_bytes >= gran);

    drop(handle);
    alloc.destroy_slab(slab).expect("destroy_slab failed");
}

#[test]
fn test_eviction_under_pressure() {
    let alloc = create_allocator();
    let gran = alloc.granularity();

    alloc.set_vram_ceiling(gran * 2);

    let slab1 = alloc.create_slab(gran * 2).expect("create slab1");
    let r1a = alloc.define_region(slab1, 0, gran).expect("define r1a");
    let r1b = alloc.define_region(slab1, gran, gran).expect("define r1b");

    let slab2 = alloc.create_slab(gran).expect("create slab2");
    let r2 = alloc.define_region(slab2, 0, gran).expect("define r2");

    let stream = std::ptr::null_mut();

    let h1a = alloc.ensure_resident(slab1, r1a, stream).expect("map r1a");
    let h1b = alloc.ensure_resident(slab1, r1b, stream).expect("map r1b");

    drop(h1a);
    drop(h1b);

    alloc.set_priority(slab2, 10).expect("set priority");

    let h2 = alloc.ensure_resident(slab2, r2, stream).expect("map r2 (should evict)");
    // SAFETY: allocator is alive.
    assert_ne!(unsafe { h2.as_ptr() }, 0);

    drop(h2);
    alloc.destroy_slab(slab2).expect("destroy slab2");
    alloc.destroy_slab(slab1).expect("destroy slab1");
}

#[test]
fn test_watermark_blocks_mapping() {
    let alloc = create_allocator();
    let gran = alloc.granularity();

    alloc.set_vram_ceiling(gran);

    let slab1 = alloc.create_slab(gran).expect("create slab1");
    let r1 = alloc.define_region(slab1, 0, gran).expect("define r1");

    let slab2 = alloc.create_slab(gran * 2).expect("create slab2");
    let r2a = alloc.define_region(slab2, 0, gran).expect("define r2a");
    let _r2b = alloc.define_region(slab2, gran, gran).expect("define r2b");

    let stream = std::ptr::null_mut();

    let h1 = alloc.ensure_resident(slab1, r1, stream).expect("map r1");

    let result = alloc.ensure_resident(slab2, r2a, stream);
    assert!(
        matches!(result, Err(VmmError::Watermarked) | Err(VmmError::NoEvictableRegions)),
        "expected Watermarked or NoEvictableRegions, got {result:?}"
    );

    drop(h1);

    alloc.set_priority(slab2, 10).expect("set priority");

    let h2a = alloc.ensure_resident(slab2, r2a, stream).expect("map r2a after promotion");
    // SAFETY: allocator is alive.
    assert_ne!(unsafe { h2a.as_ptr() }, 0);

    drop(h2a);
    alloc.destroy_slab(slab1).expect("destroy slab1");
    alloc.destroy_slab(slab2).expect("destroy slab2");
}

#[test]
fn test_refcount_prevents_eviction() {
    let alloc = create_allocator();
    let gran = alloc.granularity();

    alloc.set_vram_ceiling(gran);

    let slab = alloc.create_slab(gran * 2).expect("create slab");
    let r1 = alloc.define_region(slab, 0, gran).expect("define r1");
    let r2 = alloc.define_region(slab, gran, gran).expect("define r2");

    let stream = std::ptr::null_mut();

    let h1 = alloc.ensure_resident(slab, r1, stream).expect("map r1");

    let result = alloc.ensure_resident(slab, r2, stream);
    assert!(result.is_err(), "should fail: r1 is held and can't be evicted");

    drop(h1);

    let h2 = alloc.ensure_resident(slab, r2, stream).expect("map r2 after dropping r1");
    // SAFETY: allocator is alive.
    assert_ne!(unsafe { h2.as_ptr() }, 0);

    drop(h2);
    alloc.destroy_slab(slab).expect("destroy slab");
}

#[test]
fn test_prefetch_then_ensure() {
    let alloc = create_allocator();
    let gran = alloc.granularity();
    let slab = alloc.create_slab(gran).expect("create slab");
    let region = alloc.define_region(slab, 0, gran).expect("define region");

    alloc.prefetch(slab, region);
    std::thread::sleep(std::time::Duration::from_millis(100));

    let stream = std::ptr::null_mut();
    let handle = alloc
        .ensure_resident(slab, region, stream)
        .expect("ensure_resident after prefetch");
    // SAFETY: allocator is alive.
    assert_ne!(unsafe { handle.as_ptr() }, 0);

    drop(handle);
    alloc.destroy_slab(slab).expect("destroy slab");
}

#[test]
fn test_stats() {
    let alloc = create_allocator();
    let gran = alloc.granularity();

    let slab = alloc.create_slab(gran * 3).expect("create slab");
    alloc.define_region(slab, 0, gran).expect("define r0");
    alloc.define_region(slab, gran, gran).expect("define r1");
    alloc.define_region(slab, gran * 2, gran).expect("define r2");

    let stream = std::ptr::null_mut();
    let h0 = alloc.ensure_resident(slab, 0, stream).expect("map r0");

    let stats = alloc.stats();
    assert_eq!(stats.total_slabs, 1);
    assert_eq!(stats.total_regions, 3);
    assert!(stats.mapped_bytes >= gran);
    assert!(stats.granularity > 0);

    let slab_stats = stats.slabs[0].as_ref().unwrap();
    assert_eq!(slab_stats.regions.len(), 3);
    assert_eq!(slab_stats.regions[0].state, stagehand_vmm::RegionState::Resident);
    assert!(slab_stats.regions[0].refcount >= 1);

    drop(h0);
    alloc.destroy_slab(slab).expect("destroy slab");
}
