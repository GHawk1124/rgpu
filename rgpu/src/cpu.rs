use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::mem::MaybeUninit;

use rayon::prelude::*;

pub struct View<T> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
}

// SAFETY: a view lives only during a launch that borrows its buffer, and threads write disjoint elements.
unsafe impl<T> Send for View<T> {}
unsafe impl<T> Sync for View<T> {}

impl<T> View<T> {
    pub fn get(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Every thread gets the whole slice, like a GPU thread does, and writes only its own elements.
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Indices per task. A launch this small runs on the calling thread.
const GRAIN: usize = 256;

pub fn launch(n: [u32; 3], f: impl Fn([usize; 3]) + Sync) {
    let [x, y, z] = n.map(|v| v as usize);
    let total = x * y * z;
    (0..total.div_ceil(GRAIN)).into_par_iter().for_each(|task| {
        let start = task * GRAIN;
        let mut i = [start % x, start / x % y, start / (x * y)];
        for _ in start..total.min(start + GRAIN) {
            f(i);
            i[0] += 1;
            if i[0] == x {
                i = [0, i[1] + 1, i[2]];
                if i[1] == y {
                    i = [0, 0, i[2] + 1];
                }
            }
        }
    });
}

const ALIGN: usize = 64;

pub(crate) fn alloc(bytes: usize) -> u64 {
    let p = unsafe { alloc_zeroed(Layout::from_size_align(bytes, ALIGN).unwrap()) };
    assert!(!p.is_null(), "out of memory");
    p as u64
}

pub(crate) fn free(p: u64, bytes: usize) {
    unsafe { dealloc(p as *mut u8, Layout::from_size_align(bytes, ALIGN).unwrap()) }
}

pub(crate) fn write(p: u64, data: &[MaybeUninit<u8>]) {
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut _, data.len()) }
}

pub(crate) fn read(p: u64, out: &mut [MaybeUninit<u8>]) {
    unsafe { std::ptr::copy_nonoverlapping(p as *const _, out.as_mut_ptr(), out.len()) }
}
