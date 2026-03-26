use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::allocator::AllocatorInner;
use crate::cuda_ffi::{self, CUdeviceptr, CUevent, CUstream, CU_EVENT_DISABLE_TIMING};
use crate::slab::{RegionId, SlabId};

/// RAII guard that keeps a region resident while held. When dropped, records a
/// CUDA event on the consuming stream and decrements the region's refcount.
pub struct ResidentHandle {
    pub(crate) inner: Arc<AllocatorInner>,
    pub(crate) slab: SlabId,
    pub(crate) region: RegionId,
    pub(crate) ptr: CUdeviceptr,
    pub(crate) stream: CUstream,
}

// SAFETY: CUstream is an opaque CUDA handle. The CUDA driver API is thread-safe
// for distinct handle operations. The ResidentHandle is typically used on one thread
// but may be sent across threads (e.g., into Python GC). The underlying region
// is protected by atomic refcount.
unsafe impl Send for ResidentHandle {}

impl ResidentHandle {
    pub(crate) fn new(
        inner: Arc<AllocatorInner>,
        slab: SlabId,
        region: RegionId,
        ptr: CUdeviceptr,
        stream: CUstream,
    ) -> Self {
        Self { inner, slab, region, ptr, stream }
    }

    /// Returns the device pointer for this region's mapped memory.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only while the parent `SlabAllocator`
    /// is alive and the region remains mapped.
    #[inline]
    pub unsafe fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Returns the CUDA stream this handle was created with.
    #[inline]
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    /// Returns the slab ID for this handle.
    #[inline]
    pub fn slab_id(&self) -> SlabId {
        self.slab
    }

    /// Returns the region ID for this handle.
    #[inline]
    pub fn region_id(&self) -> RegionId {
        self.region
    }
}

impl Drop for ResidentHandle {
    fn drop(&mut self) {
        let event = create_event();

        if let Some(event) = event {
            // SAFETY: stream is a valid CUstream from ensure_resident.
            unsafe {
                let _ = cuda_ffi::cuEventRecord(event, self.stream);
            }
        }

        // Store event and decrement refcount under the per-region Mutex.
        // This ensures eviction sees the event before seeing refcount == 0.
        let slabs = match self.inner.slabs.read() {
            Ok(s) => s,
            Err(_) => {
                // RwLock poisoned — destroy event, leak refcount
                if let Some(ev) = event {
                    unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ev); }
                }
                return;
            }
        };

        if let Some(slab) = slabs.get(self.slab).and_then(|s| s.as_ref()) {
            if let Some(region) = slab.regions.get(self.region) {
                if let Ok(mut rm) = region.mutable.lock() {
                    // Destroy previous event
                    if let Some(old_event) = rm.last_use_event.take() {
                        // SAFETY: old_event from a prior ResidentHandle drop.
                        unsafe { let _ = cuda_ffi::cuEventDestroy_v2(old_event); }
                    }
                    rm.last_use_event = event;
                    // Decrement AFTER storing event
                    region.refcount.fetch_sub(1, Ordering::AcqRel);
                    return;
                }
            }
        }

        // Lookup failed — clean up event
        if let Some(ev) = event {
            unsafe { let _ = cuda_ffi::cuEventDestroy_v2(ev); }
        }
    }
}

fn create_event() -> Option<CUevent> {
    let mut event: CUevent = std::ptr::null_mut();
    // SAFETY: cuEventCreate writes to a valid local pointer.
    let result = unsafe { cuda_ffi::cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING) };
    if result == cuda_ffi::CUDA_SUCCESS {
        Some(event)
    } else {
        None
    }
}
