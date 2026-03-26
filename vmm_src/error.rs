use std::fmt;

#[derive(Debug)]
pub enum VmmError {
    CudaError(u32),
    Watermarked,
    NoEvictableRegions,
    InvalidSlab,
    InvalidRegion,
    SlabNotEmpty,
}

impl fmt::Display for VmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmmError::CudaError(code) => write!(f, "CUDA driver error: {code}"),
            VmmError::Watermarked => write!(f, "region is above watermark, use fallback"),
            VmmError::NoEvictableRegions => {
                write!(f, "no evictable regions (all have active refcounts)")
            }
            VmmError::InvalidSlab => write!(f, "invalid slab ID"),
            VmmError::InvalidRegion => write!(f, "invalid region ID"),
            VmmError::SlabNotEmpty => write!(f, "slab has active refcounts, cannot destroy"),
        }
    }
}

impl std::error::Error for VmmError {}

impl From<VmmError> for pyo3::PyErr {
    fn from(e: VmmError) -> pyo3::PyErr {
        match e {
            VmmError::Watermarked => pyo3::exceptions::PyMemoryError::new_err(
                "Region watermarked — use fallback path",
            ),
            VmmError::NoEvictableRegions => {
                pyo3::exceptions::PyMemoryError::new_err("No evictable regions — all in use")
            }
            VmmError::CudaError(code) => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {code}"))
            }
            VmmError::InvalidSlab => {
                pyo3::exceptions::PyValueError::new_err("Invalid slab ID")
            }
            VmmError::InvalidRegion => {
                pyo3::exceptions::PyValueError::new_err("Invalid region ID")
            }
            VmmError::SlabNotEmpty => {
                pyo3::exceptions::PyRuntimeError::new_err("Slab has active handles")
            }
        }
    }
}
