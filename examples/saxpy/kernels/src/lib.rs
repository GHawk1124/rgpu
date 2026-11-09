#![no_std]

#[rgpu::kernels]
pub mod kernels {
    #[kernel]
    pub fn saxpy(i: usize, a: f32, x: &[f32], y: &mut [f32]) {
        y[i] += a * x[i];
    }

    #[kernel]
    pub fn length(i: usize, x: &[f32], out: &mut [u32]) {
        out[i] = x.len() as u32;
    }

    #[kernel]
    pub fn grid(i: [usize; 2], width: u32, out: &mut [u32]) {
        let at = i[1] * width as usize + i[0];
        out[at] = at as u32;
    }
}
