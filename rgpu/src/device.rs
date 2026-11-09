#[cfg(any(rgpu = "ptx", rgpu = "nvvm"))]
pub use core::arch::nvptx as arch;

#[cfg(rgpu = "amdgpu")]
pub use core::arch::amdgpu as arch;

#[cfg(all(rgpu = "oxide", not(feature = "oxide")))]
compile_error!("kernel crates built with cuda-oxide need rgpu's `oxide` feature");

// cuda-oxide matches intrinsics by the path the kernel crate sees, so its thread functions are not re-exported.
#[cfg(rgpu = "oxide")]
pub use cuda_device::kernel;

#[cfg(rgpu = "spirv")]
pub use spirv_std::{arch, glam::UVec3, spirv};

#[cfg(any(rgpu = "ptx", rgpu = "nvvm"))]
#[inline(always)]
pub fn global_id<const X: u32, const Y: u32, const Z: u32>() -> [u32; 3] {
    use arch::*;
    unsafe {
        [
            _block_idx_x() as u32 * X + _thread_idx_x() as u32,
            _block_idx_y() as u32 * Y + _thread_idx_y() as u32,
            _block_idx_z() as u32 * Z + _thread_idx_z() as u32,
        ]
    }
}

#[cfg(rgpu = "oxide")]
#[inline(always)]
pub fn global_id<const X: u32, const Y: u32, const Z: u32>() -> [u32; 3] {
    use cuda_device::thread::*;
    [blockIdx_x() * X + threadIdx_x(), blockIdx_y() * Y + threadIdx_y(), blockIdx_z() * Z + threadIdx_z()]
}

#[cfg(rgpu = "amdgpu")]
#[inline(always)]
pub fn global_id<const X: u32, const Y: u32, const Z: u32>() -> [u32; 3] {
    use arch::*;
    [workgroup_id_x() * X + workitem_id_x(), workgroup_id_y() * Y + workitem_id_y(), workgroup_id_z() * Z + workitem_id_z()]
}

#[cfg(any(rgpu = "ptx", rgpu = "nvvm", rgpu = "amdgpu"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}
