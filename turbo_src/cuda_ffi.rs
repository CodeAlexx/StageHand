//! Raw CUDA runtime API bindings for stagehand-turbo.
//!
//! Uses the CUDA **runtime** API (libcudart) rather than the driver API
//! because we need cudaMemcpyAsync, cudaMallocHost, and stream management
//! which are simpler via the runtime API and already linked by PyTorch.

use std::ffi::c_void;

pub type CudaStream = *mut c_void;
pub type CudaEvent = *mut c_void;
pub type CudaError = u32;

pub const CUDA_SUCCESS: CudaError = 0;
pub const CUDA_ERROR_NOT_READY: CudaError = 600;

// cudaMemcpyKind
pub const MEMCPY_H2D: u32 = 1; // cudaMemcpyHostToDevice
pub const MEMCPY_D2H: u32 = 2; // cudaMemcpyDeviceToHost

// Stream flags
pub const STREAM_NON_BLOCKING: u32 = 1;

// Event flags
pub const EVENT_DISABLE_TIMING: u32 = 2;

#[link(name = "cudart")]
extern "C" {
    // Memory
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;

    // Async memcpy
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: u32,
        stream: CudaStream,
    ) -> CudaError;

    // Streams
    pub fn cudaStreamCreateWithFlags(stream: *mut CudaStream, flags: u32) -> CudaError;
    pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;

    // Events
    pub fn cudaEventCreateWithFlags(event: *mut CudaEvent, flags: u32) -> CudaError;
    pub fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> CudaError;
    pub fn cudaEventSynchronize(event: CudaEvent) -> CudaError;
    pub fn cudaEventQuery(event: CudaEvent) -> CudaError;
    pub fn cudaEventDestroy(event: CudaEvent) -> CudaError;
    pub fn cudaStreamWaitEvent(stream: CudaStream, event: CudaEvent, flags: u32) -> CudaError;

    // Device
    pub fn cudaSetDevice(device: i32) -> CudaError;
    pub fn cudaGetDevice(device: *mut i32) -> CudaError;
}

/// Convert CUDA error code to Result.
#[inline]
pub fn check(err: CudaError) -> Result<(), String> {
    if err == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA error {}", err))
    }
}
