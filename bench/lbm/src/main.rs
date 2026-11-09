//! D3Q19 FP32 BGK benchmark: every codegen backend against CUDA C++ and handwritten PTX.
//!
//! `cargo run --release -p lbm -- [size] [steps]` runs a periodic size^3 box seeded with a decaying shear wave.
//! Every variant runs the same algorithm through the same rgpu launch path, reports the best of three timed runs
//! and is checked against the analytic decay of the wave. A plain copy kernel gives the bandwidth ceiling.

use std::f64::consts::TAU;
use std::time::Instant;

use lbm_kernels::{C, kernels};
use rgpu::{Arg, Buffer, Format, Gpu, Image};

const NU: f64 = 0.1;
const U0: f64 = 0.01;
const BYTES_PER_CELL: f64 = 19.0 * 2.0 * 4.0;
const RUNS: usize = 3;

static CPP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cpp.cubin"));
static HAND: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hand.cubin"));

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let size = args.first().copied().unwrap_or(256);
    let steps = args.get(1).copied().unwrap_or(200);
    let s = size as u32;
    let cells = size * size * size;
    let omega = (1.0 / (3.0 * NU + 0.5)) as f32;
    println!("D3Q19 FP32 BGK, {size}^3 cells, {steps} steps, best of {RUNS}\n");
    println!("{:<30} {:>8} {:>8} {:>8} {:>12}", "variant", "MLUPS", "GB/s", "of copy", "decay error");

    if let Some(gpu) = Gpu::cuda(0) {
        let ptx = kernels::Module::new(&gpu, &images("ptx"));
        let ceiling = header(&gpu, size, |src, dst| ptx.copy(19 * cells, src, dst));
        for (label, codegen) in [("rust, rustc nvptx", "ptx"), ("rust, Rust-CUDA nvvm", "nvvm"), ("rust, cuda-oxide", "oxide")] {
            let images = images(codegen);
            if images.is_empty() {
                continue;
            }
            let k = kernels::Module::new(&gpu, &images);
            bench(label, &gpu, &ptx, size, steps, ceiling, |src, dst| k.step(cells, src, dst, s, s, s, omega));
        }
        for (label, data) in [("cuda c++, nvcc", CPP), ("handwritten ptx", HAND)] {
            let module = gpu.load(&[Image { codegen: "external", format: Format::Cubin, arch: env!("LBM_ARCH"), data }]);
            let step = module.kernel("step", [256, 1, 1]);
            let n = cells as u32;
            bench(label, &gpu, &ptx, size, steps, ceiling, |src, dst| {
                let args = [Arg::val(&n), src.arg(), dst.arg(), Arg::val(&s), Arg::val(&s), Arg::val(&s), Arg::val(&omega)];
                module.launch(&step, [n, 1, 1], &args);
            });
        }
    }

    if let Some(gpu) = Gpu::vulkan(0) {
        let k = kernels::load(&gpu);
        let ceiling = header(&gpu, size, |src, dst| k.copy(19 * cells, src, dst));
        bench("rust, rust-gpu spirv", &gpu, &k, size, steps, ceiling, |src, dst| k.step(cells, src, dst, s, s, s, omega));
    }

    let gpu = Gpu::cpu();
    let k = kernels::load(&gpu);
    let size = size.min(64);
    let (s, cells) = (size as u32, size * size * size);
    let ceiling = header(&gpu, size, |src, dst| k.copy(19 * cells, src, dst));
    bench("rust, cpu threads", &gpu, &k, size, 10, ceiling, |src, dst| k.step(cells, src, dst, s, s, s, omega));
}

fn images(codegen: &str) -> Vec<Image> {
    kernels::IMAGES.iter().filter(|i| i.codegen == codegen).copied().collect()
}

/// Prints the device and measures the copy bandwidth in GB/s over the same amount of memory as one step.
fn header(gpu: &Gpu, size: usize, copy: impl Fn(&Buffer<f32>, &mut Buffer<f32>)) -> f64 {
    let bytes = 19 * size * size * size * 4;
    let a = gpu.zeros::<f32>(bytes / 4);
    let mut b = gpu.zeros::<f32>(bytes / 4);
    let seconds = best(gpu, 20, || copy(&a, &mut b));
    let gbs = 2.0 * bytes as f64 / seconds / 1e9;
    println!("{} ({size}^3), copy {gbs:.1} GB/s", gpu.name());
    gbs
}

/// Seeds a shear wave with `init`, times ping-pong updates and compares the final wave amplitude to the analytic decay.
#[allow(clippy::too_many_arguments)]
fn bench(
    label: &str,
    gpu: &Gpu,
    init: &kernels::Module,
    size: usize,
    steps: usize,
    ceiling: f64,
    step: impl Fn(&Buffer<f32>, &mut Buffer<f32>),
) {
    let cells = size * size * size;
    let s = size as u32;
    let wave: Vec<f32> = (0..size).map(|y| (U0 * (TAU * y as f64 / size as f64).sin()) as f32).collect();
    let mut a = gpu.zeros::<f32>(19 * cells);
    let mut b = gpu.zeros::<f32>(19 * cells);
    init.init(cells, &mut a, &gpu.upload(&wave), s, s, s);

    let pairs = steps / 2;
    let mut swap = || {
        step(&a, &mut b);
        step(&b, &mut a);
    };
    swap();
    let seconds = best(gpu, pairs, &mut swap) / 2.0;

    let mlups = cells as f64 / seconds / 1e6;
    let gbs = mlups * BYTES_PER_CELL / 1e3;
    let t = (2 + 2 * pairs * RUNS) as f64;
    let k = TAU / size as f64;
    let expected = U0 * (-NU * k * k * t).exp();
    let error = (amplitude(&a.download(), size) - expected).abs() / expected;
    println!("  {label:<28} {mlups:>8.0} {gbs:>8.1} {:>7.0}% {error:>12.2e}", 100.0 * gbs / ceiling);
}

/// Best time of `RUNS` runs of `iterations` calls, in seconds per call.
fn best(gpu: &Gpu, iterations: usize, mut f: impl FnMut()) -> f64 {
    (0..RUNS)
        .map(|_| {
            gpu.sync();
            let start = Instant::now();
            for _ in 0..iterations {
                f();
            }
            gpu.sync();
            start.elapsed().as_secs_f64() / iterations as f64
        })
        .fold(f64::MAX, f64::min)
}

/// Projects the x velocity onto the seeded sine to get the wave amplitude.
fn amplitude(f: &[f32], size: usize) -> f64 {
    let cells = size * size * size;
    let mut sum = 0.0;
    for n in 0..cells {
        let (mut rho, mut mx) = (0.0f64, 0.0f64);
        for q in 0..19 {
            let v = f[q * cells + n] as f64;
            rho += v;
            mx += C[q][0] as f64 * v;
        }
        sum += mx / rho * (TAU * (n / size % size) as f64 / size as f64).sin();
    }
    2.0 * sum / cells as f64
}
