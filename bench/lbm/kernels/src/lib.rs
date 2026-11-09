#![no_std]

/// D3Q19 velocities in FluidX3D order.
pub const C: [[i32; 3]; 19] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, -1],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, -1],
    [0, -1, 1],
];

const A: f32 = 1.0 / 3.0;
const B: f32 = 1.0 / 18.0;
const D: f32 = 1.0 / 36.0;
const W: [f32; 19] = [A, B, B, B, B, B, B, D, D, D, D, D, D, D, D, D, D, D, D];

#[rgpu::kernels]
pub mod kernels {
    use super::{C, W};

    /// Equilibrium at rest density with a shear wave `ux = wave[y]`.
    #[kernel]
    pub fn init(i: usize, f: &mut [f32], wave: &[f32], nx: u32, ny: u32, nz: u32) {
        let cells = nx * ny * nz;
        let n = i as u32;
        let ux = wave[(n / nx % ny) as usize];
        let uu = 1.5 * ux * ux;
        for q in 0..19 {
            let cu = 3.0 * C[q][0] as f32 * ux;
            f[(q as u32 * cells + n) as usize] = W[q] * (1.0 + cu + 0.5 * cu * cu - uu);
        }
    }

    /// Streams `src` into `dst`, the bandwidth ceiling a memory bound kernel can reach.
    #[kernel]
    pub fn copy(i: usize, src: &[f32], dst: &mut [f32]) {
        dst[i] = src[i];
    }

    /// Fused stream and collide: pull from neighbours in `src` with periodic wrap, BGK relax, store to `dst`.
    #[kernel]
    pub fn step(i: usize, src: &[f32], dst: &mut [f32], nx: u32, ny: u32, nz: u32, omega: f32) {
        let cells = nx * ny * nz;
        let n = i as u32;
        let xs = around(n % nx, nx);
        let ys = around(n / nx % ny, ny);
        let zs = around(n / (nx * ny), nz);

        let mut f = [0.0f32; 19];
        unroll!(q in 0..19 {
            let [cx, cy, cz] = C[q];
            let s = xs[(cx + 1) as usize] + nx * (ys[(cy + 1) as usize] + ny * zs[(cz + 1) as usize]);
            f[q] = src[(q as u32 * cells + s) as usize];
        });

        let mut rho = 0.0;
        let mut u = [0.0f32; 3];
        unroll!(q in 0..19 {
            rho += f[q];
            u[0] += C[q][0] as f32 * f[q];
            u[1] += C[q][1] as f32 * f[q];
            u[2] += C[q][2] as f32 * f[q];
        });
        let u = [u[0] / rho, u[1] / rho, u[2] / rho];
        let uu = 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

        unroll!(q in 0..19 {
            let cu = 3.0 * (C[q][0] as f32 * u[0] + C[q][1] as f32 * u[1] + C[q][2] as f32 * u[2]);
            let feq = W[q] * rho * (1.0 + cu + 0.5 * cu * cu - uu);
            dst[(q as u32 * cells + n) as usize] = f[q] + omega * (feq - f[q]);
        });
    }

    /// Coordinates a velocity component of -1, 0 or 1 pulls from, with periodic wrap.
    fn around(v: u32, n: u32) -> [u32; 3] {
        [if v + 1 == n { 0 } else { v + 1 }, v, if v == 0 { n - 1 } else { v - 1 }]
    }
}
