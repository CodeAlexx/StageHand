use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::allocator::{AllocatorInner, SlabAllocator};
use crate::cuda_ffi::CUstream;
use crate::dlpack::{self, DLDevice, DLManagedTensor, DLPackContext, DLTensor, KDLCUDA};
use crate::error::VmmError;
use crate::handle::ResidentHandle;
use crate::slab::{RegionId, SlabId};

/// PyCapsule destructor called if the capsule is GC'd without being consumed
/// by torch.from_dlpack(). This prevents leaking the DLManagedTensor and
/// DLPackContext when from_dlpack raises an exception.
///
/// # Safety
///
/// `capsule` must be a valid PyCapsule containing a `*mut DLManagedTensor`
/// allocated via `Box::into_raw`.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr()) };
    if !ptr.is_null() {
        // Capsule still named "dltensor" means it was NOT consumed.
        // Call the DLPack deleter to free the tensor and context.
        let managed = ptr as *mut DLManagedTensor;
        unsafe { dlpack::dlpack_deleter(managed); }
    } else {
        // PyCapsule_GetPointer sets a Python error when the name doesn't
        // match ("PyCapsule_GetPointer called with incorrect name"). This
        // is expected when PyTorch consumed the capsule (renamed it to
        // "used_dltensor"). Clear the error so it doesn't propagate to the
        // caller — PyO3 would otherwise interpret it as a real exception.
        unsafe { pyo3::ffi::PyErr_Clear(); }
    }
}

/// Python-visible SlabAllocator.
#[pyclass(name = "SlabAllocator")]
pub struct PySlabAllocator {
    alloc: SlabAllocator,
}

#[pymethods]
impl PySlabAllocator {
    #[new]
    #[pyo3(signature = (device=0, ceiling_mb=None))]
    fn new(device: i32, ceiling_mb: Option<usize>) -> PyResult<Self> {
        let ceiling_bytes = ceiling_mb.map(|mb| mb * 1024 * 1024);
        let alloc = SlabAllocator::new(device, ceiling_bytes)?;
        Ok(Self { alloc })
    }

    fn create_slab(&self, total_size: usize) -> PyResult<usize> {
        Ok(self.alloc.create_slab(total_size)?)
    }

    fn define_region(&self, slab: usize, offset: usize, size: usize) -> PyResult<usize> {
        Ok(self.alloc.define_region(slab, offset, size)?)
    }

    /// Returns a ResidentHandle. The region stays mapped while the handle exists.
    ///
    /// `stream` is a raw CUDA stream pointer as an integer.
    /// Get it from `torch.cuda.current_stream().cuda_stream`.
    fn ensure_resident(
        &self,
        slab: usize,
        region: usize,
        stream: usize,
    ) -> PyResult<PyResidentHandle> {
        let stream_ptr = stream as CUstream;
        let handle = self.alloc.ensure_resident(slab, region, stream_ptr)?;
        Ok(PyResidentHandle {
            handle: Some(handle),
            allocator: Arc::clone(&self.alloc.inner),
            slab,
            region,
        })
    }

    fn prefetch(&self, slab: usize, region: usize) -> PyResult<()> {
        self.alloc.prefetch(slab, region);
        Ok(())
    }

    fn set_priority(&self, slab: usize, priority: u32) -> PyResult<()> {
        Ok(self.alloc.set_priority(slab, priority)?)
    }

    fn set_vram_ceiling(&self, ceiling_bytes: usize) -> PyResult<()> {
        self.alloc.set_vram_ceiling(ceiling_bytes);
        Ok(())
    }

    fn destroy_slab(&self, slab: usize) -> PyResult<()> {
        Ok(self.alloc.destroy_slab(slab)?)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.alloc.stats();
        let dict = PyDict::new(py);
        dict.set_item("total_slabs", s.total_slabs)?;
        dict.set_item("total_regions", s.total_regions)?;
        dict.set_item("mapped_bytes", s.mapped_bytes)?;
        dict.set_item("vram_ceiling", s.vram_ceiling)?;
        dict.set_item("granularity", s.granularity)?;
        Ok(dict.into())
    }
}

/// Python-visible ResidentHandle.
///
/// SAFETY for Sync: PyResidentHandle contains a ResidentHandle which has a CUstream
/// (*mut c_void). The CUstream is only used in the Drop impl (to record an event)
/// and is never shared for concurrent reads. The PyO3 GIL ensures Python methods
/// are serialized.
#[pyclass(name = "ResidentHandle")]
pub struct PyResidentHandle {
    handle: Option<ResidentHandle>,
    allocator: Arc<AllocatorInner>,
    slab: SlabId,
    region: RegionId,
}

// SAFETY: CUstream is an opaque CUDA handle. PyResidentHandle methods are
// serialized by the Python GIL. The CUstream is only accessed in Drop
// (to record an event) which is also GIL-serialized in Python.
unsafe impl Sync for PyResidentHandle {}

#[pymethods]
impl PyResidentHandle {
    /// Returns a PyTorch tensor backed by VMM-mapped VRAM.
    ///
    /// The tensor has its own refcount via DLPack — the region stays mapped
    /// until both the handle AND all tensors derived from it are GC'd.
    #[pyo3(signature = (dtype, shape))]
    fn as_tensor(&self, py: Python<'_>, dtype: &str, shape: Vec<i64>) -> PyResult<PyObject> {
        let handle = self.handle.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Handle already released")
        })?;

        let dl_dtype = dlpack::parse_dtype(dtype).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Unsupported dtype: {dtype}"))
        })?;

        // Create a NEW ResidentHandle for the DLPack context so the tensor
        // has its own independent refcount.
        let dlpack_handle = self.allocator_ensure_resident(handle.stream())?;

        // SAFETY: handle is alive so the allocator is alive and region is mapped.
        let data_ptr = unsafe { handle.as_ptr() } as *mut std::ffi::c_void;

        let ctx = Box::new(DLPackContext {
            _handle: dlpack_handle,
            shape,
        });

        // SAFETY (DLPack shape lifetime): shape_ptr points into ctx.shape (a Vec
        // inside the Box<DLPackContext>).  PyTorch's from_dlpack copies the shape
        // into its own internal storage during tensor creation and never dereferences
        // DLTensor.shape again afterwards.  The DLPackContext (and its shape Vec)
        // remains alive until the DLPack deleter fires when PyTorch GC's the tensor,
        // so the pointer cannot dangle while PyTorch might use it.
        let shape_ptr = ctx.shape.as_ptr() as *mut i64;
        let ndim = ctx.shape.len() as i32;

        let dl_tensor = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: KDLCUDA,
                    device_id: self.allocator.device_ordinal,
                },
                ndim,
                dtype: dl_dtype,
                shape: shape_ptr,
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: Box::into_raw(ctx) as *mut std::ffi::c_void,
            deleter: Some(dlpack::dlpack_deleter),
        });

        let raw_ptr = Box::into_raw(dl_tensor);

        // Create PyCapsule named "dltensor" — the DLPack protocol contract.
        // The capsule destructor handles cleanup if from_dlpack fails:
        // PyTorch renames consumed capsules to "used_dltensor", so the
        // destructor only fires for unconsumed capsules.
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                raw_ptr as *mut std::ffi::c_void,
                c"dltensor".as_ptr(),
                Some(capsule_destructor),
            )
        };
        if capsule.is_null() {
            // PyCapsule_New failed — destructor was NOT registered, clean up manually.
            // SAFETY: raw_ptr was just created via Box::into_raw
            unsafe {
                let tensor_box = Box::from_raw(raw_ptr);
                if !tensor_box.manager_ctx.is_null() {
                    let _ = Box::from_raw(tensor_box.manager_ctx as *mut DLPackContext);
                }
            }
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create DLPack capsule",
            ));
        }

        // SAFETY: capsule is a valid new reference from PyCapsule_New.
        let capsule_obj: Bound<'_, pyo3::PyAny> =
            unsafe { Bound::from_owned_ptr(py, capsule) };

        // Call torch.from_dlpack(capsule) to get a PyTorch tensor
        let torch = py.import("torch")?;
        let tensor = torch.call_method1("from_dlpack", (&capsule_obj,))?;

        Ok(tensor.into())
    }

    /// Explicitly release the handle. Decrements refcount.
    /// Tensors already created via as_tensor() remain valid.
    fn release(&mut self) -> PyResult<()> {
        if self.handle.take().is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Handle already released",
            ));
        }
        Ok(())
    }

    fn __del__(&mut self) -> PyResult<()> {
        // Drop handle if not already released
        self.handle.take();
        Ok(())
    }
}

impl PyResidentHandle {
    /// Create a new ResidentHandle for the same region (for DLPack context).
    fn allocator_ensure_resident(
        &self,
        stream: CUstream,
    ) -> Result<ResidentHandle, VmmError> {
        let slabs = self.allocator.slabs.read().map_err(|_| VmmError::InvalidSlab)?;
        let slab = slabs.get(self.slab).and_then(|s| s.as_ref()).ok_or(VmmError::InvalidSlab)?;
        let region = slab.regions.get(self.region).ok_or(VmmError::InvalidRegion)?;

        // Region must be Resident (our handle keeps it alive)
        if region.load_state() != crate::slab::RegionState::Resident {
            return Err(VmmError::InvalidRegion);
        }

        region.refcount.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let ptr = slab.base_ptr + region.offset as u64;

        Ok(ResidentHandle::new(
            Arc::clone(&self.allocator),
            self.slab,
            self.region,
            ptr,
            stream,
        ))
    }
}
