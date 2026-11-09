use rgpu::{Format, Gpu, Image};
use saxpy_kernels::kernels;

fn gpus() -> Vec<Gpu> {
    [Gpu::cuda(0), Gpu::hip(0), Gpu::vulkan(0), Some(Gpu::cpu())].into_iter().flatten().collect()
}

fn saxpy(gpu: &Gpu, images: &[Image], n: usize) {
    let k = kernels::Module::new(gpu, images);
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = gpu.upload(&x);
    let mut y = gpu.upload(&vec![1.0f32; n]);
    k.saxpy(n, 2.0, &x, &mut y);
    let y = y.download();
    assert!(y.iter().enumerate().all(|(i, &v)| v == 2.0 * i as f32 + 1.0), "{}", gpu.name());
}

#[test]
fn every_backend() {
    for gpu in gpus() {
        for n in [0, 100_003] {
            saxpy(&gpu, kernels::IMAGES, n);
        }
    }
}

#[test]
fn slice_lengths_match_the_host() {
    for gpu in gpus() {
        let k = kernels::load(&gpu);
        for len in [0, 1, 3, 1000] {
            let x = gpu.zeros::<f32>(len);
            let mut out = gpu.upload(&[u32::MAX]);
            k.length(1, &x, &mut out);
            assert_eq!(out.download(), [len as u32], "{}", gpu.name());
        }
    }
}

#[test]
fn grid_covers_every_cell_once() {
    for gpu in gpus() {
        let k = kernels::load(&gpu);
        let [w, h] = [37, 19];
        let mut out = gpu.upload(&vec![u32::MAX; w * h]);
        k.grid([w, h], w as u32, &mut out);
        assert!(out.download().iter().enumerate().all(|(i, &v)| v == i as u32), "{}", gpu.name());
    }
}

#[test]
fn zeros_reach_every_byte() {
    for gpu in gpus() {
        for len in 1..9 {
            drop(gpu.upload(&vec![u8::MAX; len]));
            assert!(gpu.zeros::<u8>(len).download().iter().all(|&b| b == 0), "{} len {len}", gpu.name());
        }
    }
}

#[test]
fn ptx_jit_fallback() {
    let Some(gpu) = Gpu::cuda(0) else { return };
    let ptx: Vec<Image> = kernels::IMAGES.iter().filter(|i| i.format == Format::Ptx).copied().collect();
    saxpy(&gpu, &ptx, 4096);
}

#[test]
fn modules_are_shareable() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Gpu>();
    send_sync::<kernels::Module>();
    send_sync::<rgpu::Buffer<f32>>();
}
