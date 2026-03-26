//! stagehand-vmm: Zero-overhead GPU memory residency via CUDA Virtual Memory Management.
//!
//! Phase 2 — Rust core with PyO3 Python bindings and DLPack tensor wrapping.
//!
//! # Architecture
//!
//! - **Slab**: Reserved virtual address range on the GPU. Contains regions.
//! - **Region**: A chunk of a slab that can be independently mapped/unmapped.
//! - **SlabAllocator**: Central manager. All methods take `&self` (thread-safe).
//! - **ResidentHandle**: RAII guard. Keeps a region mapped while alive.
//! - **DLPack**: Zero-copy tensor protocol. `as_tensor()` returns a PyTorch tensor.
//!
//! The fast path (already-Resident regions) uses only an RwLock read lock and
//! atomic operations — no Mutex, no CUDA calls. The cold path (mapping, eviction)
//! takes a Mutex and performs CUDA VMM calls.

#[allow(non_snake_case)]
pub mod cuda_ffi;
pub mod error;
pub mod slab;
pub mod allocator;
pub mod eviction;
pub mod prefetch;
pub mod handle;
pub mod dlpack;
pub mod python;

pub use allocator::{AllocatorStats, RegionStats, SlabAllocator, SlabStats};
pub use error::VmmError;
pub use handle::ResidentHandle;
pub use slab::{RegionId, RegionState, SlabId};

use pyo3::prelude::*;

#[pymodule]
fn stagehand_vmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PySlabAllocator>()?;
    m.add_class::<python::PyResidentHandle>()?;
    Ok(())
}
