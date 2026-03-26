use crate::handle::ResidentHandle;

/// DLPack device type for CUDA.
pub const KDLCUDA: i32 = 2;

#[repr(C)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Context stored alongside the DLManagedTensor. Dropping this drops the
/// ResidentHandle, which decrements the refcount and allows eviction.
pub(crate) struct DLPackContext {
    pub(crate) _handle: ResidentHandle,
    pub(crate) shape: Vec<i64>,
}

/// DLPack deleter callback. Called by PyTorch when the tensor's PyCapsule is GC'd.
///
/// # Safety
///
/// `tensor` must point to a `DLManagedTensor` that was allocated via `Box::into_raw`.
/// `manager_ctx` must point to a `DLPackContext` that was allocated via `Box::into_raw`.
/// This function is called exactly once by the DLPack consumer (PyTorch).
pub unsafe extern "C" fn dlpack_deleter(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    // SAFETY: tensor was allocated by Box::into_raw in python.rs as_tensor.
    let tensor = unsafe { Box::from_raw(tensor) };
    if !tensor.manager_ctx.is_null() {
        // SAFETY: manager_ctx was allocated by Box::into_raw in python.rs as_tensor.
        let _ctx = unsafe { Box::from_raw(tensor.manager_ctx as *mut DLPackContext) };
        // _ctx drops here, dropping the ResidentHandle, decrementing refcount
    }
}

/// Parse a dtype string to a DLDataType.
pub fn parse_dtype(dtype: &str) -> Option<DLDataType> {
    match dtype {
        "float32" => Some(DLDataType { code: 2, bits: 32, lanes: 1 }),
        "float16" => Some(DLDataType { code: 2, bits: 16, lanes: 1 }),
        "bfloat16" => Some(DLDataType { code: 4, bits: 16, lanes: 1 }),
        "float8_e4m3fn" => Some(DLDataType { code: 6, bits: 8, lanes: 1 }),
        "int8" => Some(DLDataType { code: 0, bits: 8, lanes: 1 }),
        "uint8" => Some(DLDataType { code: 1, bits: 8, lanes: 1 }),
        _ => None,
    }
}
