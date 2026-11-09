use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::{cpu, cuda, hip, vulkan};

/// Plain data a buffer can hold.
///
/// # Safety
///
/// Every bit pattern of `size_of::<Self>()` bytes must be a valid `Self`, since device memory holds whatever kernels wrote.
pub unsafe trait Pod: Copy {}

macro_rules! pod {
    ($($t:ty)*) => {
        $(unsafe impl Pod for $t {})*
    };
}

pod!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

#[derive(Clone, Copy, Debug)]
pub struct Image {
    pub codegen: &'static str,
    pub format: Format,
    pub arch: &'static str,
    pub data: &'static [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Ptx,
    Cubin,
    Spirv,
    Amdgpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Hip,
    Vulkan,
    Cpu,
}

#[derive(Clone)]
pub struct Gpu(Arc<Device>);

#[allow(clippy::large_enum_variant, reason = "it lives behind an Arc")]
enum Device {
    Cuda(cuda::Device),
    Hip(hip::Device),
    Vulkan(vulkan::Device),
    Cpu,
}

pub(crate) enum Raw {
    Ptr(u64),
    Vk(vulkan::Buf),
}

impl Default for Gpu {
    fn default() -> Gpu {
        Gpu::new()
    }
}

impl Gpu {
    /// Uses `RGPU_BACKEND` (cuda, hip, vulkan or cpu) when set, otherwise the first of CUDA, HIP, Vulkan and CPU that works.
    pub fn new() -> Gpu {
        match std::env::var("RGPU_BACKEND").as_deref() {
            Ok("cuda") => Gpu::cuda(0).expect("no CUDA device"),
            Ok("hip") => Gpu::hip(0).expect("no HIP device"),
            Ok("vulkan") => Gpu::vulkan(0).expect("no Vulkan device"),
            Ok("cpu") => Gpu::cpu(),
            Ok(other) => panic!("unknown RGPU_BACKEND {other}"),
            Err(_) => Gpu::cuda(0).or_else(|| Gpu::hip(0)).or_else(|| Gpu::vulkan(0)).unwrap_or_else(Gpu::cpu),
        }
    }

    pub fn cuda(ordinal: usize) -> Option<Gpu> {
        cuda::Device::new(ordinal).map(|d| Gpu(Arc::new(Device::Cuda(d))))
    }

    pub fn hip(ordinal: usize) -> Option<Gpu> {
        hip::Device::new(ordinal).map(|d| Gpu(Arc::new(Device::Hip(d))))
    }

    /// Discrete GPUs come first in the ordering.
    pub fn vulkan(index: usize) -> Option<Gpu> {
        vulkan::Device::new(index).map(|d| Gpu(Arc::new(Device::Vulkan(d))))
    }

    pub fn cpu() -> Gpu {
        Gpu(Arc::new(Device::Cpu))
    }

    pub fn backend(&self) -> Backend {
        match &*self.0 {
            Device::Cuda(_) => Backend::Cuda,
            Device::Hip(_) => Backend::Hip,
            Device::Vulkan(_) => Backend::Vulkan,
            Device::Cpu => Backend::Cpu,
        }
    }

    pub fn name(&self) -> String {
        match &*self.0 {
            Device::Cuda(d) => d.name(),
            Device::Hip(d) => d.name(),
            Device::Vulkan(d) => d.name(),
            Device::Cpu => "cpu".into(),
        }
    }

    pub fn sync(&self) {
        match &*self.0 {
            Device::Cuda(d) => d.sync(),
            Device::Hip(d) => d.sync(),
            Device::Vulkan(d) => d.sync(),
            Device::Cpu => {}
        }
    }

    pub fn upload<T: Pod>(&self, data: &[T]) -> Buffer<T> {
        let mut b = Buffer { raw: self.alloc(size_of_val(data)), len: data.len(), gpu: self.clone(), _t: PhantomData };
        b.upload(data);
        b
    }

    pub fn zeros<T: Pod>(&self, len: usize) -> Buffer<T> {
        let bytes = len * size_of::<T>();
        let raw = self.alloc(bytes);
        match (&*self.0, &raw) {
            (Device::Cuda(d), Raw::Ptr(p)) => d.zero(*p, bytes),
            (Device::Hip(d), Raw::Ptr(p)) => d.zero(*p, bytes),
            (Device::Vulkan(d), Raw::Vk(b)) => d.zero(b),
            _ => {}
        }
        Buffer { raw, len, gpu: self.clone(), _t: PhantomData }
    }

    /// Loads the best image for this device: an exact binary, then a compatible one, then IR the driver compiles.
    pub fn load(&self, images: &[Image]) -> Module {
        let raw = match &*self.0 {
            Device::Cuda(d) => RawModule::Cuda(d.load(images)),
            Device::Hip(d) => RawModule::Hip(d.load(images)),
            Device::Vulkan(d) => RawModule::Vulkan(d.load(images)),
            Device::Cpu => RawModule::Cpu,
        };
        Module { raw, gpu: self.clone() }
    }

    fn alloc(&self, bytes: usize) -> Raw {
        // No backend allows an empty allocation.
        let bytes = bytes.max(1);
        match &*self.0 {
            Device::Cuda(d) => Raw::Ptr(d.alloc(bytes)),
            Device::Hip(d) => Raw::Ptr(d.alloc(bytes)),
            Device::Vulkan(d) => Raw::Vk(d.alloc(bytes)),
            Device::Cpu => Raw::Ptr(cpu::alloc(bytes)),
        }
    }
}

pub struct Buffer<T> {
    raw: Raw,
    len: usize,
    gpu: Gpu,
    _t: PhantomData<T>,
}

impl<T: Pod> Buffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn upload(&mut self, data: &[T]) {
        assert_eq!(data.len(), self.len, "upload length mismatch");
        if data.is_empty() {
            return;
        }
        // SAFETY: `MaybeUninit<u8>` can hold any byte, padding included.
        let bytes: &[MaybeUninit<u8>] = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) };
        match (&*self.gpu.0, &self.raw) {
            (Device::Cuda(d), Raw::Ptr(p)) => d.write(*p, bytes),
            (Device::Hip(d), Raw::Ptr(p)) => d.write(*p, bytes),
            (Device::Vulkan(d), Raw::Vk(b)) => d.write(b, bytes),
            (Device::Cpu, Raw::Ptr(p)) => cpu::write(*p, bytes),
            _ => unreachable!(),
        }
    }

    pub fn download(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len);
        if self.len == 0 {
            return out;
        }
        let spare = &mut out.spare_capacity_mut()[..self.len];
        let bytes: &mut [MaybeUninit<u8>] = unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast(), size_of_val(spare)) };
        match (&*self.gpu.0, &self.raw) {
            (Device::Cuda(d), Raw::Ptr(p)) => d.read(*p, bytes),
            (Device::Hip(d), Raw::Ptr(p)) => d.read(*p, bytes),
            (Device::Vulkan(d), Raw::Vk(b)) => d.read(b, bytes),
            (Device::Cpu, Raw::Ptr(p)) => cpu::read(*p, bytes),
            _ => unreachable!(),
        }
        // SAFETY: every byte was written and any bytes are a valid `T: Pod`.
        unsafe { out.set_len(self.len) };
        out
    }

    #[doc(hidden)]
    pub fn view(&self) -> cpu::View<T> {
        let Raw::Ptr(p) = self.raw else { unreachable!() };
        cpu::View { ptr: p as *mut T, len: self.len }
    }

    #[doc(hidden)]
    pub fn arg(&self) -> Arg<'_> {
        Arg(ArgKind::Buf(&self.raw, &self.len))
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        match (&*self.gpu.0, &self.raw) {
            (Device::Cuda(d), Raw::Ptr(p)) => d.free(*p),
            (Device::Hip(d), Raw::Ptr(p)) => d.free(*p),
            (Device::Vulkan(d), Raw::Vk(b)) => d.free(b),
            (Device::Cpu, Raw::Ptr(p)) => cpu::free(*p, (self.len * size_of::<T>()).max(1)),
            _ => unreachable!(),
        }
    }
}

#[doc(hidden)]
pub struct Arg<'a>(pub(crate) ArgKind<'a>);

pub(crate) enum ArgKind<'a> {
    Val(*const u8, usize, usize),
    Buf(&'a Raw, &'a usize),
}

impl Arg<'_> {
    pub fn val<T: Copy>(v: &T) -> Arg<'_> {
        Arg(ArgKind::Val((v as *const T).cast(), size_of::<T>(), align_of::<T>()))
    }
}

pub(crate) fn params(args: &[Arg]) -> [*mut c_void; 64] {
    let mut params = [std::ptr::null_mut(); 64];
    let mut n = 0;
    for a in args {
        match a.0 {
            ArgKind::Val(p, ..) => {
                params[n] = p as *mut c_void;
                n += 1;
            }
            ArgKind::Buf(Raw::Ptr(p), len) => {
                params[n] = p as *const u64 as *mut c_void;
                params[n + 1] = len as *const usize as *mut c_void;
                n += 2;
            }
            ArgKind::Buf(..) => unreachable!(),
        }
    }
    params
}

pub trait Dims {
    fn dims(self) -> [u32; 3];
}

impl Dims for usize {
    fn dims(self) -> [u32; 3] {
        [self, 1, 1].dims()
    }
}

impl Dims for [usize; 2] {
    fn dims(self) -> [u32; 3] {
        [self[0], self[1], 1].dims()
    }
}

impl Dims for [usize; 3] {
    fn dims(self) -> [u32; 3] {
        self.map(|n| u32::try_from(n).expect("launch size exceeds u32"))
    }
}

pub struct Module {
    raw: RawModule,
    gpu: Gpu,
}

enum RawModule {
    Cuda(cuda::Module),
    Hip(hip::Module),
    Vulkan(vulkan::Module),
    Cpu,
}

pub struct Kernel {
    raw: RawKernel,
    block: [u32; 3],
    gpu: Gpu,
}

enum RawKernel {
    Cuda(cuda::Function),
    Hip(hip::Function),
    Vulkan(vulkan::Kernel),
    Cpu,
}

impl Module {
    pub fn kernel(&self, name: &str, block: [u32; 3]) -> Kernel {
        let raw = match (&*self.gpu.0, &self.raw) {
            (Device::Cuda(d), RawModule::Cuda(m)) => RawKernel::Cuda(d.function(m, name)),
            (Device::Hip(d), RawModule::Hip(m)) => RawKernel::Hip(d.function(m, name)),
            (Device::Vulkan(_), RawModule::Vulkan(_)) => RawKernel::Vulkan(vulkan::Kernel::new(name)),
            _ => RawKernel::Cpu,
        };
        Kernel { raw, block, gpu: self.gpu.clone() }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.raw, RawModule::Cpu)
    }

    pub fn launch(&self, kernel: &Kernel, n: [u32; 3], args: &[Arg]) {
        if n.contains(&0) {
            return;
        }
        let block = kernel.block;
        let grid = [0, 1, 2].map(|i| n[i].div_ceil(block[i]));
        match (&*self.gpu.0, &self.raw, &kernel.raw) {
            (Device::Cuda(d), _, RawKernel::Cuda(f)) => d.launch(f, grid, block, args),
            (Device::Hip(d), _, RawKernel::Hip(f)) => d.launch(f, grid, block, args),
            (Device::Vulkan(d), RawModule::Vulkan(m), RawKernel::Vulkan(k)) => d.launch(m, k, grid, args),
            _ => unreachable!(),
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        match (&*self.gpu.0, &self.raw) {
            (Device::Cuda(d), RawModule::Cuda(m)) => d.unload(m),
            (Device::Hip(d), RawModule::Hip(m)) => d.unload(m),
            (Device::Vulkan(d), RawModule::Vulkan(m)) => d.unload(m),
            _ => {}
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let (Device::Vulkan(d), RawKernel::Vulkan(k)) = (&*self.gpu.0, &self.raw) {
            d.destroy(k)
        }
    }
}
