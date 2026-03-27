//! stagehand-turbo: Rust transfer engine for Stagehand block-swap training.
//!
//! Moves the hot path (CPU↔GPU weight transfers) into Rust with dedicated
//! CUDA streams, bypassing Python dispatch and GIL contention. Python calls
//! `engine.step(block_idx, direction)`, Rust handles all DMA internally.

mod cuda_ffi;
mod engine;

use engine::TransferEngine;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "TransferEngine")]
struct PyTransferEngine {
    inner: TransferEngine,
}

#[pymethods]
impl PyTransferEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: TransferEngine::new(),
        }
    }

    /// Register a block's parameters. Returns block index.
    /// param_sizes: list of byte sizes for each parameter.
    fn register_block(&mut self, param_sizes: Vec<usize>) -> PyResult<usize> {
        Ok(self.inner.register_block(param_sizes))
    }

    /// Allocate pinned CPU + GPU memory.
    /// window_size: max blocks on GPU simultaneously (determines GPU alloc).
    /// Call after all register_block() calls.
    fn initialize(&mut self, device: i32, window_size: usize) -> PyResult<()> {
        self.inner
            .initialize(device, window_size)
            .map_err(PyRuntimeError::new_err)
    }

    /// Copy initial weight data from CPU tensor data_ptrs into pinned pool.
    /// src_ptrs: list of int (tensor.data_ptr() for each param, on CPU).
    fn stage_block(&mut self, block_idx: usize, src_ptrs: Vec<usize>) -> PyResult<()> {
        self.inner
            .stage_block_to_pinned(block_idx, &src_ptrs)
            .map_err(PyRuntimeError::new_err)
    }

    /// Hot path: ensure block is on GPU, async evict prev, async prefetch next.
    /// direction: "forward" or "backward".
    /// Releases GIL during CUDA synchronization.
    fn step(&mut self, py: Python, block_idx: usize, direction: &str) -> PyResult<()> {
        // Validate direction before releasing GIL
        if direction != "forward" && direction != "backward" {
            return Err(PyValueError::new_err(
                "direction must be 'forward' or 'backward'",
            ));
        }
        let engine = &mut self.inner;
        // Release GIL during CUDA sync + async dispatch
        py.allow_threads(|| engine.step(block_idx, direction))
            .map_err(PyRuntimeError::new_err)
    }

    /// Force-evict a block to CPU pinned memory (blocking).
    fn evict_sync(&mut self, block_idx: usize) -> PyResult<()> {
        self.inner
            .async_evict(block_idx)
            .map_err(PyRuntimeError::new_err)?;
        self.inner.sync_evict().map_err(PyRuntimeError::new_err)
    }

    /// Get GPU data_ptr for a parameter (only valid while block is on GPU).
    fn gpu_ptr(&self, block_idx: usize, param_idx: usize) -> PyResult<usize> {
        self.inner
            .gpu_ptr(block_idx, param_idx)
            .map_err(PyRuntimeError::new_err)
    }

    /// Get pinned CPU data_ptr for a parameter.
    fn cpu_ptr(&self, block_idx: usize, param_idx: usize) -> PyResult<usize> {
        self.inner
            .cpu_ptr(block_idx, param_idx)
            .map_err(PyRuntimeError::new_err)
    }

    /// Set GPU data_ptrs for a slot from pre-allocated torch tensors.
    fn set_slot_gpu_ptrs(&mut self, slot_idx: usize, ptrs: Vec<usize>) -> PyResult<()> {
        self.inner.set_slot_gpu_ptrs(slot_idx, &ptrs).map_err(PyRuntimeError::new_err)
    }

    fn block_slot(&self, block_idx: usize) -> Option<usize> { self.inner.block_slot(block_idx) }
    fn num_blocks(&self) -> usize { self.inner.num_blocks() }
    fn num_gpu_slots(&self) -> usize { self.inner.num_gpu_slots() }
    fn num_free_slots(&self) -> usize { self.inner.num_free_slots() }

    /// Current location of a block: "cpu", "gpu", "h2d", "d2h".
    fn block_location(&self, block_idx: usize) -> PyResult<String> {
        self.inner
            .block_location(block_idx)
            .map(|s| s.to_string())
            .map_err(PyRuntimeError::new_err)
    }

    /// Drain all pending transfers. Call before shutdown.
    fn drain(&mut self) -> PyResult<()> {
        self.inner.sync_evict().map_err(PyRuntimeError::new_err)?;
        // Also sync prefetch stream
        unsafe {
            cuda_ffi::check(cuda_ffi::cudaStreamSynchronize(
                self.inner.stream_prefetch,
            ))
            .map_err(PyRuntimeError::new_err)?;
        }
        Ok(())
    }
}

/// Python module definition.
#[pymodule]
fn stagehand_turbo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransferEngine>()?;
    Ok(())
}
