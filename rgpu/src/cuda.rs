use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::mem::MaybeUninit;
use std::ptr::{null, null_mut};

use crate::gpu::{Arg, Format, Image, params};

api! {
    cuInit(c_uint);
    cuDeviceGet(*mut c_int, c_int);
    cuDeviceGetAttribute(*mut c_int, c_int, c_int);
    cuDeviceGetName(*mut c_char, c_int, c_int);
    cuDevicePrimaryCtxRetain(*mut *mut c_void, c_int);
    cuDevicePrimaryCtxRelease_v2(c_int);
    cuCtxSetCurrent(*mut c_void);
    cuCtxSynchronize();
    cuMemAlloc_v2(*mut u64, usize);
    cuMemFree_v2(u64);
    cuMemcpyHtoD_v2(u64, *const c_void, usize);
    cuMemcpyDtoH_v2(*mut c_void, u64, usize);
    cuMemsetD8_v2(u64, u8, usize);
    cuModuleLoadData(*mut *mut c_void, *const c_void);
    cuModuleUnload(*mut c_void);
    cuModuleGetFunction(*mut *mut c_void, *mut c_void, *const c_char);
    cuLaunchKernel(*mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void);
    cuGetErrorName(c_int, *mut *const c_char);
}

const LIBS: &[&str] = if cfg!(windows) { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };

pub struct Device {
    api: Api,
    ctx: *mut c_void,
    dev: c_int,
    /// Compute capability as `major * 10 + minor`.
    arch: u32,
}

// SAFETY: the driver API is thread safe and every call first makes `ctx` current on the calling thread.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

pub struct Module(*mut c_void);
pub struct Function(*mut c_void);

// SAFETY: module and function handles are plain driver handles, valid from any thread with the context current.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Device {
    pub fn new(ordinal: usize) -> Option<Device> {
        let api = Api::load(LIBS)?;
        let (mut dev, mut major, mut minor, mut ctx) = (0, 0, 0, null_mut());
        unsafe {
            if (api.cuInit)(0) != 0 || (api.cuDeviceGet)(&mut dev, ordinal as c_int) != 0 {
                return None;
            }
            (api.cuDeviceGetAttribute)(&mut major, 75, dev);
            (api.cuDeviceGetAttribute)(&mut minor, 76, dev);
            if (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev) != 0 {
                return None;
            }
        }
        Some(Device { api, ctx, dev, arch: (major * 10 + minor) as u32 })
    }

    fn check(&self, r: c_int) {
        if r != 0 {
            let mut name = null();
            unsafe { (self.api.cuGetErrorName)(r, &mut name) };
            let name = if name.is_null() { r.to_string() } else { unsafe { CStr::from_ptr(name) }.to_string_lossy().into() };
            panic!("CUDA error {name}");
        }
    }

    fn bind(&self) {
        call!(self, cuCtxSetCurrent(self.ctx));
    }

    pub fn name(&self) -> String {
        let mut name = [0 as c_char; 256];
        call!(self, cuDeviceGetName(name.as_mut_ptr(), 256, self.dev));
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
        format!("{name} (sm_{})", self.arch)
    }

    pub fn sync(&self) {
        self.bind();
        call!(self, cuCtxSynchronize());
    }

    pub fn alloc(&self, bytes: usize) -> u64 {
        let mut p = 0;
        self.bind();
        call!(self, cuMemAlloc_v2(&mut p, bytes));
        p
    }

    pub fn free(&self, p: u64) {
        self.bind();
        call!(self, cuMemFree_v2(p));
    }

    pub fn zero(&self, p: u64, bytes: usize) {
        self.bind();
        call!(self, cuMemsetD8_v2(p, 0, bytes));
    }

    pub fn write(&self, p: u64, data: &[MaybeUninit<u8>]) {
        self.bind();
        call!(self, cuMemcpyHtoD_v2(p, data.as_ptr().cast(), data.len()));
    }

    pub fn read(&self, p: u64, out: &mut [MaybeUninit<u8>]) {
        self.bind();
        call!(self, cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), p, out.len()));
    }

    pub fn load(&self, images: &[Image]) -> Module {
        let sm = |i: &&Image| i.arch.trim_start_matches("sm_").parse::<u32>().unwrap_or(u32::MAX);
        let cubin =
            images.iter().filter(|i| i.format == Format::Cubin && sm(i) / 10 == self.arch / 10 && sm(i) <= self.arch).max_by_key(sm);
        let ptx = images.iter().filter(|i| i.format == Format::Ptx && sm(i) <= self.arch).max_by_key(sm);
        let image = cubin.or(ptx).unwrap_or_else(|| panic!("no CUDA image for sm_{}", self.arch));
        let mut m = null_mut();
        self.bind();
        call!(self, cuModuleLoadData(&mut m, image.data.as_ptr().cast()));
        Module(m)
    }

    pub fn unload(&self, m: &Module) {
        self.bind();
        call!(self, cuModuleUnload(m.0));
    }

    pub fn function(&self, m: &Module, name: &str) -> Function {
        let name = CString::new(name).unwrap();
        let mut f = null_mut();
        self.bind();
        call!(self, cuModuleGetFunction(&mut f, m.0, name.as_ptr()));
        Function(f)
    }

    pub fn launch(&self, f: &Function, grid: [u32; 3], block: [u32; 3], args: &[Arg]) {
        let mut params = params(args);
        let [gx, gy, gz] = grid;
        let [bx, by, bz] = block;
        self.bind();
        call!(self, cuLaunchKernel(f.0, gx, gy, gz, bx, by, bz, 0, null_mut(), params.as_mut_ptr(), null_mut()));
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        call!(self, cuDevicePrimaryCtxRelease_v2(self.dev));
    }
}
