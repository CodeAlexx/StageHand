//! Transfer engine: the core of stagehand-turbo.
//!
//! Owns CUDA streams, pinned memory pool, and a GPU slot pool.
//! All transfers run on dedicated streams, GIL-free.
//!
//! GPU memory is allocated for a WINDOW of blocks (not all blocks).
//! Blocks rotate through GPU slots as they're prefetched and evicted.

use std::ffi::c_void;
use std::ptr;

use crate::cuda_ffi::*;

/// Per-parameter in a GPU slot: a pre-allocated GPU buffer.
#[derive(Clone)]
pub struct GpuSlot {
    pub ptrs: Vec<*mut c_void>,   // GPU pointers, one per param
    pub sizes: Vec<usize>,        // byte sizes, one per param
    pub assigned_to: Option<usize>, // which block index, or None if free
}

unsafe impl Send for GpuSlot {}
unsafe impl Sync for GpuSlot {}

/// Per-parameter metadata for a block.
#[derive(Clone)]
pub struct ParamInfo {
    pub size_bytes: usize,
    pub cpu_ptr: *mut c_void,  // pinned host memory (always allocated)
}

unsafe impl Send for ParamInfo {}
unsafe impl Sync for ParamInfo {}

/// Per-block metadata.
pub struct BlockSlot {
    pub params: Vec<ParamInfo>,
    pub total_bytes: usize,
    pub location: Location,
    pub gpu_slot_idx: Option<usize>,  // which GPU slot this block occupies
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Location {
    Cpu,
    Gpu,
    InTransitH2D,
    InTransitD2H,
}

pub struct TransferEngine {
    pub blocks: Vec<BlockSlot>,
    pub gpu_slots: Vec<GpuSlot>,     // pool of GPU memory slots
    pub free_slots: Vec<usize>,       // indices of unoccupied GPU slots
    pub stream_prefetch: CudaStream,
    pub stream_evict: CudaStream,
    pub event_prefetch: CudaEvent,
    pub event_evict: CudaEvent,
    pub device: i32,
    initialized: bool,
}

unsafe impl Send for TransferEngine {}
unsafe impl Sync for TransferEngine {}

impl TransferEngine {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            gpu_slots: Vec::new(),
            free_slots: Vec::new(),
            stream_prefetch: ptr::null_mut(),
            stream_evict: ptr::null_mut(),
            event_prefetch: ptr::null_mut(),
            event_evict: ptr::null_mut(),
            device: 0,
            initialized: false,
        }
    }

    /// Register a block. Returns block index.
    pub fn register_block(&mut self, param_sizes: Vec<usize>) -> usize {
        let total: usize = param_sizes.iter().sum();
        let params = param_sizes
            .into_iter()
            .map(|sz| ParamInfo {
                size_bytes: sz,
                cpu_ptr: ptr::null_mut(),
            })
            .collect();
        let idx = self.blocks.len();
        self.blocks.push(BlockSlot {
            params,
            total_bytes: total,
            location: Location::Cpu,
            gpu_slot_idx: None,
        });
        idx
    }

    /// Allocate resources. `window_size` = max blocks on GPU simultaneously.
    pub fn initialize(&mut self, device: i32, window_size: usize) -> Result<(), String> {
        if self.initialized {
            return Err("Already initialized".into());
        }
        if self.blocks.is_empty() {
            return Err("No blocks registered".into());
        }
        self.device = device;

        // Find the largest block to size GPU slots
        let max_params = self.blocks.iter().map(|b| b.params.len()).max().unwrap_or(0);
        let param_max_sizes: Vec<usize> = (0..max_params)
            .map(|pi| {
                self.blocks.iter()
                    .filter(|b| pi < b.params.len())
                    .map(|b| b.params[pi].size_bytes)
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        unsafe {
            check(cudaSetDevice(device))?;

            // Create streams
            check(cudaStreamCreateWithFlags(&mut self.stream_prefetch, STREAM_NON_BLOCKING))?;
            check(cudaStreamCreateWithFlags(&mut self.stream_evict, STREAM_NON_BLOCKING))?;
            check(cudaEventCreateWithFlags(&mut self.event_prefetch, EVENT_DISABLE_TIMING))?;
            check(cudaEventCreateWithFlags(&mut self.event_evict, EVENT_DISABLE_TIMING))?;

            // Allocate pinned CPU memory for ALL blocks (fits in 64GB RAM)
            for block in &mut self.blocks {
                for param in &mut block.params {
                    if param.size_bytes == 0 { continue; }
                    check(cudaMallocHost(&mut param.cpu_ptr, param.size_bytes))?;
                }
            }

            // Create empty GPU slots — Python will fill in the data_ptrs
            // from pre-allocated torch tensors via set_slot_gpu_ptrs().
            let actual_window = window_size.min(self.blocks.len());
            for _slot_idx in 0..actual_window {
                let slot_ptrs = vec![ptr::null_mut(); max_params];
                let slot_sizes = param_max_sizes.clone();
                self.free_slots.push(self.gpu_slots.len());
                self.gpu_slots.push(GpuSlot {
                    ptrs: slot_ptrs,
                    sizes: slot_sizes,
                    assigned_to: None,
                });
            }
        }
        self.initialized = true;
        Ok(())
    }

    /// Copy initial weights from CPU tensors into pinned pool.
    pub fn stage_block_to_pinned(&mut self, block_idx: usize, src_ptrs: &[usize]) -> Result<(), String> {
        let block = &self.blocks[block_idx];
        if src_ptrs.len() != block.params.len() {
            return Err(format!("Expected {} ptrs, got {}", block.params.len(), src_ptrs.len()));
        }
        for (i, param) in block.params.iter().enumerate() {
            if param.size_bytes == 0 { continue; }
            unsafe {
                ptr::copy_nonoverlapping(
                    src_ptrs[i] as *const u8,
                    param.cpu_ptr as *mut u8,
                    param.size_bytes,
                );
            }
        }
        Ok(())
    }

    /// Acquire a GPU slot for a block. Returns slot index.
    fn acquire_gpu_slot(&mut self, block_idx: usize) -> Result<usize, String> {
        // Check if block already has a slot
        if let Some(slot_idx) = self.blocks[block_idx].gpu_slot_idx {
            return Ok(slot_idx);
        }
        // Get a free slot
        if let Some(slot_idx) = self.free_slots.pop() {
            self.gpu_slots[slot_idx].assigned_to = Some(block_idx);
            self.blocks[block_idx].gpu_slot_idx = Some(slot_idx);
            Ok(slot_idx)
        } else {
            Err("No free GPU slots — increase window_size or evict a block".into())
        }
    }

    /// Release a block's GPU slot back to the free pool.
    fn release_gpu_slot(&mut self, block_idx: usize) {
        if let Some(slot_idx) = self.blocks[block_idx].gpu_slot_idx.take() {
            self.gpu_slots[slot_idx].assigned_to = None;
            self.free_slots.push(slot_idx);
        }
    }

    /// Issue async H2D for a block.
    pub fn async_prefetch(&mut self, block_idx: usize) -> Result<(), String> {
        if block_idx >= self.blocks.len() { return Ok(()); }
        let loc = self.blocks[block_idx].location;
        if loc == Location::Gpu || loc == Location::InTransitH2D { return Ok(()); }
        if loc == Location::InTransitD2H {
            unsafe { check(cudaStreamSynchronize(self.stream_evict))?; }
            // Slot was released during evict — need to re-acquire
        }

        let slot_idx = self.acquire_gpu_slot(block_idx)?;
        let block = &self.blocks[block_idx];
        let slot = &self.gpu_slots[slot_idx];

        unsafe {
            for (i, param) in block.params.iter().enumerate() {
                if param.size_bytes == 0 { continue; }
                check(cudaMemcpyAsync(
                    slot.ptrs[i],
                    param.cpu_ptr as *const c_void,
                    param.size_bytes,
                    MEMCPY_H2D,
                    self.stream_prefetch,
                ))?;
            }
            check(cudaEventRecord(self.event_prefetch, self.stream_prefetch))?;
        }
        self.blocks[block_idx].location = Location::InTransitH2D;
        Ok(())
    }

    /// Issue async D2H for a block.
    pub fn async_evict(&mut self, block_idx: usize) -> Result<(), String> {
        if block_idx >= self.blocks.len() { return Ok(()); }
        let loc = self.blocks[block_idx].location;
        if loc == Location::Cpu || loc == Location::InTransitD2H { return Ok(()); }

        let slot_idx = match self.blocks[block_idx].gpu_slot_idx {
            Some(s) => s,
            None => return Ok(()), // no GPU slot, nothing to evict
        };

        let block = &self.blocks[block_idx];
        let slot = &self.gpu_slots[slot_idx];

        unsafe {
            // Wait for compute on default stream before copying back
            check(cudaEventRecord(self.event_evict, ptr::null_mut()))?;
            check(cudaStreamWaitEvent(self.stream_evict, self.event_evict, 0))?;
            for (i, param) in block.params.iter().enumerate() {
                if param.size_bytes == 0 { continue; }
                check(cudaMemcpyAsync(
                    param.cpu_ptr,
                    slot.ptrs[i] as *const c_void,
                    param.size_bytes,
                    MEMCPY_D2H,
                    self.stream_evict,
                ))?;
            }
        }
        self.blocks[block_idx].location = Location::InTransitD2H;
        // Release slot AFTER eviction completes (deferred to sync_evict)
        Ok(())
    }

    /// Wait for prefetch to land. Mark block as GPU-resident.
    pub fn sync_prefetch(&mut self, block_idx: usize) -> Result<(), String> {
        if block_idx >= self.blocks.len() { return Err("Block out of range".into()); }
        if self.blocks[block_idx].location == Location::Gpu { return Ok(()); }
        if self.blocks[block_idx].location != Location::InTransitH2D {
            self.async_prefetch(block_idx)?;
        }
        unsafe {
            check(cudaEventSynchronize(self.event_prefetch))?;
            // Make the default stream (where PyTorch compute runs) wait on the
            // prefetch event. Without this, GPU compute can read the slot buffer
            // before the H2D transfer has actually landed.
            check(cudaStreamWaitEvent(ptr::null_mut(), self.event_prefetch, 0))?;
        }
        self.blocks[block_idx].location = Location::Gpu;
        Ok(())
    }

    /// Drain eviction stream. Finalize any pending D2H and release GPU slots.
    pub fn sync_evict(&mut self) -> Result<(), String> {
        unsafe { check(cudaStreamSynchronize(self.stream_evict))?; }
        // Release GPU slots for all blocks that were evicting
        for idx in 0..self.blocks.len() {
            if self.blocks[idx].location == Location::InTransitD2H {
                self.blocks[idx].location = Location::Cpu;
                self.release_gpu_slot(idx);
            }
        }
        Ok(())
    }

    /// Blocking evict: async_evict + sync.
    pub fn evict_sync(&mut self, block_idx: usize) -> Result<(), String> {
        self.async_evict(block_idx)?;
        self.sync_evict()
    }

    /// Hot path: ensure block on GPU, async evict prev, async prefetch next.
    pub fn step(&mut self, block_idx: usize, direction: &str) -> Result<(), String> {
        let n = self.blocks.len();
        if block_idx >= n { return Err(format!("block_idx {} >= {}", block_idx, n)); }

        // 1. Sync: ensure this block is on GPU
        self.sync_prefetch(block_idx)?;

        // 2. Determine evict/prefetch targets
        let (evict_idx, prefetch_idx) = match direction {
            "forward" => (
                if block_idx > 0 { Some(block_idx - 1) } else { None },
                if block_idx + 1 < n { Some(block_idx + 1) } else { None },
            ),
            "backward" => (
                if block_idx + 1 < n { Some(block_idx + 1) } else { None },
                if block_idx > 0 { Some(block_idx - 1) } else { None },
            ),
            _ => return Err(format!("Bad direction: {}", direction)),
        };

        // 3. Evict previous (async, runs on stream_evict)
        if let Some(idx) = evict_idx {
            // Must sync evict stream first to free the GPU slot
            self.sync_evict()?;
            self.async_evict(idx)?;
        }

        // 4. Prefetch next (async, runs on stream_prefetch)
        if let Some(idx) = prefetch_idx {
            // Need a free slot — sync evict if none available
            if self.free_slots.is_empty() && self.blocks[idx].gpu_slot_idx.is_none() {
                self.sync_evict()?;
            }
            let _ = self.async_prefetch(idx); // ok to fail if no slot
        }

        Ok(())
    }

    /// Get GPU data_ptr for a parameter (only valid when block is on GPU).
    pub fn gpu_ptr(&self, block_idx: usize, param_idx: usize) -> Result<usize, String> {
        let block = &self.blocks[block_idx];
        let slot_idx = block.gpu_slot_idx.ok_or("Block has no GPU slot")?;
        let slot = &self.gpu_slots[slot_idx];
        if param_idx >= slot.ptrs.len() { return Err("Param out of range".into()); }
        Ok(slot.ptrs[param_idx] as usize)
    }

    /// Get pinned CPU data_ptr for a parameter (always valid).
    pub fn cpu_ptr(&self, block_idx: usize, param_idx: usize) -> Result<usize, String> {
        if block_idx >= self.blocks.len() { return Err("Block out of range".into()); }
        let block = &self.blocks[block_idx];
        if param_idx >= block.params.len() { return Err("Param out of range".into()); }
        Ok(block.params[param_idx].cpu_ptr as usize)
    }

    /// Set GPU data_ptrs for a slot (from pre-allocated torch tensors).
    pub fn set_slot_gpu_ptrs(&mut self, slot_idx: usize, ptrs: &[usize]) -> Result<(), String> {
        if slot_idx >= self.gpu_slots.len() {
            return Err("Slot index out of range".into());
        }
        let slot = &mut self.gpu_slots[slot_idx];
        for (i, &ptr) in ptrs.iter().enumerate() {
            if i < slot.ptrs.len() {
                slot.ptrs[i] = ptr as *mut c_void;
            }
        }
        Ok(())
    }

    /// Which GPU slot does a block occupy? None if not on GPU.
    pub fn block_slot(&self, block_idx: usize) -> Option<usize> {
        if block_idx >= self.blocks.len() { return None; }
        self.blocks[block_idx].gpu_slot_idx
    }

    pub fn num_blocks(&self) -> usize { self.blocks.len() }
    pub fn num_gpu_slots(&self) -> usize { self.gpu_slots.len() }
    pub fn num_free_slots(&self) -> usize { self.free_slots.len() }

    pub fn block_location(&self, block_idx: usize) -> Result<&str, String> {
        if block_idx >= self.blocks.len() { return Err("Out of range".into()); }
        Ok(match self.blocks[block_idx].location {
            Location::Cpu => "cpu",
            Location::Gpu => "gpu",
            Location::InTransitH2D => "h2d",
            Location::InTransitD2H => "d2h",
        })
    }
}

impl Drop for TransferEngine {
    fn drop(&mut self) {
        unsafe {
            // Free pinned CPU memory (Rust owns these via cudaMallocHost)
            for block in &mut self.blocks {
                for param in &mut block.params {
                    if !param.cpu_ptr.is_null() {
                        let _ = cudaFreeHost(param.cpu_ptr);
                        param.cpu_ptr = ptr::null_mut();
                    }
                }
            }
            // GPU slot pointers are NOT ours — they belong to PyTorch tensors
            // allocated in Python (torch.empty). PyTorch's CUDA caching
            // allocator manages their lifecycle. Calling cudaFree here would
            // corrupt the allocator and cause NaN/segfaults.
            for slot in &mut self.gpu_slots {
                for ptr in &mut slot.ptrs {
                    *ptr = ptr::null_mut();
                }
            }
            if !self.stream_prefetch.is_null() { let _ = cudaStreamDestroy(self.stream_prefetch); }
            if !self.stream_evict.is_null() { let _ = cudaStreamDestroy(self.stream_evict); }
            if !self.event_prefetch.is_null() { let _ = cudaEventDestroy(self.event_prefetch); }
            if !self.event_evict.is_null() { let _ = cudaEventDestroy(self.event_evict); }
        }
    }
}
