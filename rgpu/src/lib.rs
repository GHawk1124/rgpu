#![cfg_attr(rgpu, no_std)]
#![cfg_attr(rgpu, feature(core_float_math))]
#![cfg_attr(rgpu, allow(internal_features))]
#![cfg_attr(any(rgpu = "ptx", rgpu = "nvvm"), feature(stdarch_nvptx, core_intrinsics))]
#![cfg_attr(rgpu = "amdgpu", feature(stdarch_amdgpu, core_intrinsics))]

pub use rgpu_macros::{kernels, unroll};

#[cfg(rgpu)]
pub mod device;

#[cfg(not(rgpu))]
macro_rules! api {
    ($($f:ident($($t:ty),*);)*) => {
        #[allow(non_snake_case)]
        struct Api {
            $($f: unsafe extern "C" fn($($t),*) -> std::ffi::c_int,)*
            _lib: libloading::Library,
        }

        impl Api {
            fn load(names: &[&str]) -> Option<Api> {
                let lib = names.iter().find_map(|n| unsafe { libloading::Library::new(*n) }.ok())?;
                unsafe {
                    Some(Api {
                        $($f: *lib.get(concat!(stringify!($f), "\0").as_bytes()).ok()?,)*
                        _lib: lib,
                    })
                }
            }
        }
    };
}

#[cfg(not(rgpu))]
macro_rules! call {
    ($d:expr, $f:ident($($a:expr),*)) => {
        $d.check(unsafe { ($d.api.$f)($($a),*) })
    };
}

#[cfg(not(rgpu))]
pub mod cpu;
#[cfg(not(rgpu))]
mod cuda;
#[cfg(not(rgpu))]
mod gpu;
#[cfg(not(rgpu))]
mod hip;
#[cfg(not(rgpu))]
mod vulkan;
#[cfg(not(rgpu))]
pub use gpu::*;

pub mod prelude {
    pub use crate::{Float, unroll};
}

/// Float math that works the same on the host and every device backend.
pub trait Float {
    fn sqrt(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

macro_rules! float {
    ($t:ident) => {
        impl Float for $t {
            #[inline(always)]
            fn sqrt(self) -> $t {
                #[cfg(not(rgpu))]
                return $t::sqrt(self);
                #[cfg(rgpu)]
                return core::$t::math::sqrt(self);
            }

            #[inline(always)]
            fn mul_add(self, a: $t, b: $t) -> $t {
                #[cfg(not(rgpu))]
                return $t::mul_add(self, a, b);
                #[cfg(rgpu)]
                return core::$t::math::mul_add(self, a, b);
            }
        }
    };
}

float!(f32);
float!(f64);
