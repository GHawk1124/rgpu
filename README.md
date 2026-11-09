# rgpu

Write a GPU kernel once in plain Rust. rgpu compiles it with every Rust GPU codegen backend, embeds all the binaries in your executable and runs it on whatever GPU the machine has, through CUDA, HIP, Vulkan or CPU threads.

```rust
// kernels/src/lib.rs
#![no_std]

#[rgpu::kernels]
pub mod kernels {
    #[kernel]
    pub fn saxpy(i: usize, a: f32, x: &[f32], y: &mut [f32]) {
        y[i] += a * x[i];
    }
}
```

```rust
// kernels/build.rs
fn main() {
    rgpu_build::build()
}
```

```toml
# kernels/Cargo.toml
[dependencies]
rgpu = { path = "../rgpu" }

[build-dependencies]
rgpu-build = { path = "../rgpu-build" }
```

```rust
// src/main.rs
use my_kernels::kernels;

fn main() {
    let gpu = rgpu::Gpu::new();
    let k = kernels::load(&gpu);
    let x = gpu.upload(&[1.0f32; 1024]);
    let mut y = gpu.zeros::<f32>(1024);
    k.saxpy(1024, 2.0, &x, &mut y);
    assert_eq!(y.download()[0], 2.0);
}
```

A kernel is the body of a parallel loop, the same closure you would hand to rayon. Its first parameter is the index (`usize`, `[usize; 2]` or `[usize; 3]`) and the launch runs it once per index. The rest are `&[T]`, `&mut [T]` or `Copy` values. `examples/saxpy` is a complete project.

Buffers hold `rgpu::Pod` types, the ones every bit pattern is valid for: integers, floats and arrays of them. A `#[repr(C)]` struct of such fields opts in with `unsafe impl rgpu::Pod for MyStruct {}`.

## Backends

| codegen | output | runtime | toolchain | default |
|---|---|---|---|---|
| rustc nvptx | PTX, cubins | CUDA driver | nightly-2026-08-28 | on |
| [rust-gpu](https://github.com/Rust-GPU/rust-gpu) 0.10 | SPIR-V | Vulkan | nightly-2026-07-03 | on |
| rustc amdgpu | code objects | HIP | nightly-2026-08-28 | on |
| [Rust-CUDA](https://github.com/Rust-GPU/rust-cuda) NVVM | PTX, cubins | CUDA driver | nightly-2026-04-02 | `nvvm: true` |
| [cuda-oxide](https://github.com/NVIDIA/cuda-rust) | PTX, cubins | CUDA driver | nightly-2026-08-28 | `oxide: true` and rgpu's `oxide` feature |
| none | host code | CPU threads | stable | always |

Your crate stays on stable Rust. `rgpu-build` installs each pinned nightly with rustup on first use, builds the codegen backends it needs once into a per user cache (`%LOCALAPPDATA%\rgpu`, `~/.cache/rgpu` or `RGPU_CACHE`) and compiles the kernel crate once per backend into `target/rgpu`.

At runtime `Gpu::new()` takes `RGPU_BACKEND` (`cuda`, `hip`, `vulkan`, `cpu`) if set, otherwise the first of CUDA, HIP, Vulkan and CPU that works. Loading picks the best embedded binary and falls back to compiling on the fly:

- CUDA: a cubin for the exact arch, then one of the same major version, then PTX the driver JIT compiles.
- AMD: each code object in build order, exact targets first, then the generic ones (`gfx11-generic` and so on).
- Vulkan: the driver compiles the SPIR-V.

## Configuration

`rgpu_build::build()` uses the defaults. Set any field of `Builder` to change them:

```rust
rgpu_build::Builder { nvvm: true, amdgpu: vec!["gfx1100".into(), "gfx11-generic".into()], cubins: vec![80, 86, 89, 90], ..Default::default() }.build()
```

| field | default | |
|---|---|---|
| `ptx`, `nvvm`, `oxide`, `spirv` | on, off, off, on | codegens to build |
| `amdgpu` | all `gfx*-generic` targets | AMD targets, in load order |
| `ptx_arch` | 75 | lowest compute capability of the PTX |
| `cubins` | the GPUs in this machine | archs to precompile with `ptxas` |
| `fma` | on | fuse multiply and add like nvcc does |

Going deeper:

- `#[kernel(block = 128)]` or `#[kernel(block = [16, 16])]` sets the block size. The default is 256, 16x16 or 8x8x4.
- `rgpu::device::arch` is the backend's own intrinsics module under `#[cfg(rgpu = "ptx")]` and the like.
- `kernels::Module::new(&gpu, images)` loads a chosen subset of `kernels::IMAGES`.
- `gpu.load(images)` and `module.launch(..)` run kernels written in other languages that follow the kernel ABI below.

## Writing kernels

Kernel crates are `#![no_std]` since they compile for GPUs. On the host they compile normally and back the CPU fallback.

- Float math beyond `+ - * /` goes through `rgpu::Float` (`sqrt`, `mul_add`), which the kernels module imports for you.
- `unroll!(q in 0..19 { .. })` repeats a body with a constant index, like `#pragma unroll`. rust-gpu keeps plain loops rolled, which puts any local array they index into slow memory.
- rust-gpu cannot compile everything in `core` yet, `[T; N]::map` for example. If a kernel builds for PTX but not SPIR-V, that is the likely cause.
- Writes through `&mut [T]` from many threads follow GPU rules: each index must be written by one thread. The CPU fallback relies on the same rule.
- `x.len()` is the buffer's length on every backend.

## Kernel ABI

Each kernel compiles to an entry named after the function taking, in order: the launch size as one `u32` per dimension, then each parameter, with a slice passed as its address followed by its length as `u64`. On Vulkan, slices bind to descriptor set 0 in order, each with its byte length as the range, and values pack into push constants with C layout. `bench/lbm/cuda` has a CUDA C++ and a handwritten PTX kernel that load through this ABI.

## Requirements

- rustup.
- Windows: Visual Studio 2022 with the C++ tools. NVVM needs 17.13 or newer.
- Linux: a C++ compiler.
- CUDA toolkit for cubins (`ptxas`) and NVVM (`libnvvm`). Without it only PTX is embedded and the driver JIT compiles it.
- At runtime only the GPU driver: CUDA, the HIP runtime or a Vulkan driver.

## Notes

- cuda-oxide officially supports Linux and Ampere or newer. Through rgpu it also builds on Windows and runs on Turing: its device code is compiled for an x86_64 Linux object, since it embeds a PTX copy only into ELF, and rgpu keeps the PTX it writes beside it.
- Vulkan records launches into one command buffer and submits it on `sync`, a download, an upload or a free. CUDA and HIP start each launch right away.
- The CPU fallback splits a launch into tasks of 256 indices on rayon's thread pool. A launch that small runs on the calling thread.
- The HIP runtime is written against the HIP API but untested, since no AMD GPU was available.

## Benchmark

`cargo run --release -p lbm` runs a D3Q19 FP32 BGK lattice Boltzmann step (fused pull streaming and collision, periodic, SoA) on a 256^3 grid. All variants use the same algorithm and launch path and are checked against the analytic decay of a shear wave. `LBM_NVVM=1` and `LBM_OXIDE=1` add those codegens.

RTX 2070 (sm_75), Windows 11, CUDA 13.0, 256^3 cells, best of 3, desktop running on the same GPU:

| variant | MLUPS | GB/s | of copy bandwidth |
|---|---|---|---|
| rust, rustc nvptx | 2379 | 361.5 | 96% |
| rust, Rust-CUDA nvvm | 2382 | 362.1 | 96% |
| rust, cuda-oxide | 2379 | 361.6 | 96% |
| cuda c++, nvcc | 2350 | 357.2 | 95% |
| handwritten ptx | 2349 | 357.1 | 95% |
| rust, rust-gpu spirv (Vulkan) | 2361 | 358.9 | 97% |
| rust, cpu threads (64^3) | 64-76 | 9.7-11.6 | 70-80% |

The kernel is memory bound, so every backend reaches the bandwidth ceiling (a plain copy kernel, 376 GB/s here) and the generated code costs nothing over nvcc or handwritten PTX. Two things were needed to get there and are now defaults or tools in rgpu:

- rustc's LLVM did not fully unroll the 19 iteration loops and kept the distribution array in local memory, at about a quarter of the speed. `rgpu-build` raises the NVPTX unroll threshold, which nvcc and LLVM's own AMDGPU backend do in effect.
- rust-gpu passes loops to the driver rolled and ran at about half speed until the kernel used `unroll!`.

AMD code objects build for every generic target with no scratch memory but could not be run here.

## Layout

- `rgpu`: runtime (CUDA, HIP, Vulkan, CPU) and device intrinsics.
- `rgpu-macros`: `#[kernels]` and `unroll!`.
- `rgpu-build`: the build script helper.
- `examples/saxpy`: minimal project and backend tests.
- `bench/lbm`: the benchmark, which needs nvcc. `cargo build` and `cargo test` leave it out.
