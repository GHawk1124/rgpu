use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::mem::MaybeUninit;
use std::ptr::null_mut;

use crate::gpu::{Arg, Format, Image, params};

api! {
    hipInit(c_uint);
    hipGetDeviceCount(*mut c_int);
    hipSetDevice(c_int);
    hipDeviceGetName(*mut c_char, c_int, c_int);
    hipDeviceSynchronize();
    hipMalloc(*mut u64, usize);
    hipFree(u64);
    hipMemcpyHtoD(u64, *const c_void, usize);
    hipMemcpyDtoH(*mut c_void, u64, usize);
    hipMemset(u64, c_int, usize);
    hipModuleLoadData(*mut *mut c_void, *const c_void);
    hipModuleUnload(*mut c_void);
    hipModuleGetFunction(*mut *mut c_void, *mut c_void, *const c_char);
    hipModuleLaunchKernel(*mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void);
}

const LIBS: &[&str] = if cfg!(windows) {
    &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
} else {
    &["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"]
};

pub struct Device {
    api: Api,
    dev: c_int,
}

pub struct Module(*mut c_void);
pub struct Function(*mut c_void);

// SAFETY: module and function handles are plain driver handles, valid from any thread.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Device {
    pub fn new(ordinal: usize) -> Option<Device> {
        let api = Api::load(LIBS)?;
        let mut count = 0;
        unsafe {
            if (api.hipInit)(0) != 0 || (api.hipGetDeviceCount)(&mut count) != 0 || ordinal >= count as usize {
                return None;
            }
        }
        Some(Device { api, dev: ordinal as c_int })
    }

    fn check(&self, r: c_int) {
        assert!(r == 0, "HIP error {r}");
    }

    fn bind(&self) {
        call!(self, hipSetDevice(self.dev));
    }

    pub fn name(&self) -> String {
        let mut name = [0 as c_char; 256];
        call!(self, hipDeviceGetName(name.as_mut_ptr(), 256, self.dev));
        unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into()
    }

    pub fn sync(&self) {
        self.bind();
        call!(self, hipDeviceSynchronize());
    }

    pub fn alloc(&self, bytes: usize) -> u64 {
        let mut p = 0;
        self.bind();
        call!(self, hipMalloc(&mut p, bytes));
        p
    }

    pub fn free(&self, p: u64) {
        self.bind();
        call!(self, hipFree(p));
    }

    pub fn zero(&self, p: u64, bytes: usize) {
        self.bind();
        call!(self, hipMemset(p, 0, bytes));
    }

    pub fn write(&self, p: u64, data: &[MaybeUninit<u8>]) {
        self.bind();
        call!(self, hipMemcpyHtoD(p, data.as_ptr().cast(), data.len()));
    }

    pub fn read(&self, p: u64, out: &mut [MaybeUninit<u8>]) {
        self.bind();
        call!(self, hipMemcpyDtoH(out.as_mut_ptr().cast(), p, out.len()));
    }

    pub fn load(&self, images: &[Image]) -> Module {
        self.bind();
        images
            .iter()
            .filter(|i| i.format == Format::Amdgpu)
            .find_map(|i| {
                let mut m = null_mut();
                (unsafe { (self.api.hipModuleLoadData)(&mut m, i.data.as_ptr().cast()) } == 0).then_some(Module(m))
            })
            .expect("no AMDGPU image for this device")
    }

    pub fn unload(&self, m: &Module) {
        self.bind();
        call!(self, hipModuleUnload(m.0));
    }

    pub fn function(&self, m: &Module, name: &str) -> Function {
        let name = CString::new(name).unwrap();
        let mut f = null_mut();
        self.bind();
        call!(self, hipModuleGetFunction(&mut f, m.0, name.as_ptr()));
        Function(f)
    }

    pub fn launch(&self, f: &Function, grid: [u32; 3], block: [u32; 3], args: &[Arg]) {
        let mut params = params(args);
        let [gx, gy, gz] = grid;
        let [bx, by, bz] = block;
        self.bind();
        call!(self, hipModuleLaunchKernel(f.0, gx, gy, gz, bx, by, bz, 0, null_mut(), params.as_mut_ptr(), null_mut()));
    }
}
