use saxpy_kernels::kernels;

fn main() {
    let gpu = rgpu::Gpu::new();
    let k = kernels::load(&gpu);

    let n = 1 << 20;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = gpu.upload(&x);
    let mut y = gpu.upload(&vec![1.0f32; n]);

    k.saxpy(n, 2.0, &x, &mut y);

    let y = y.download();
    assert!(y.iter().enumerate().all(|(i, &v)| v == 2.0 * i as f32 + 1.0));
    println!("{}: ok", gpu.name());
}
