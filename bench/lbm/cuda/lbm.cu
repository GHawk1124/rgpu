// D3Q19 BGK fused stream and collide in idiomatic CUDA C++.
// Same algorithm and kernel ABI as the Rust `step`: pull streaming with periodic wrap, SoA layout f[q * cells + n].

#include <cstddef>

__device__ __forceinline__ void around(unsigned v, unsigned n, unsigned out[3]) {
    out[0] = v + 1 == n ? 0 : v + 1;
    out[1] = v;
    out[2] = v == 0 ? n - 1 : v - 1;
}

extern "C" __global__ void __launch_bounds__(256) step(unsigned count, const float* __restrict__ src, size_t, float* __restrict__ dst,
                                                       size_t, unsigned nx, unsigned ny, unsigned nz, float omega) {
    const int c[19][3] = {{0, 0, 0},  {1, 0, 0},   {-1, 0, 0}, {0, 1, 0},  {0, -1, 0}, {0, 0, 1},  {0, 0, -1},
                          {1, 1, 0},  {-1, -1, 0}, {1, 0, 1},  {-1, 0, -1}, {0, 1, 1},  {0, -1, -1}, {1, -1, 0},
                          {-1, 1, 0}, {1, 0, -1},  {-1, 0, 1}, {0, 1, -1}, {0, -1, 1}};
    const float w[19] = {1.f / 3,  1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 36, 1.f / 36, 1.f / 36,
                         1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36};

    unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= count) return;
    unsigned cells = nx * ny * nz;
    unsigned xs[3], ys[3], zs[3];
    around(n % nx, nx, xs);
    around(n / nx % ny, ny, ys);
    around(n / (nx * ny), nz, zs);

    float f[19];
#pragma unroll
    for (int q = 0; q < 19; q++) {
        unsigned s = xs[c[q][0] + 1] + nx * (ys[c[q][1] + 1] + ny * zs[c[q][2] + 1]);
        f[q] = src[q * cells + s];
    }

    float rho = 0, u[3] = {0, 0, 0};
#pragma unroll
    for (int q = 0; q < 19; q++) {
        rho += f[q];
#pragma unroll
        for (int d = 0; d < 3; d++) u[d] += c[q][d] * f[q];
    }
    for (int d = 0; d < 3; d++) u[d] /= rho;
    float uu = 1.5f * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

#pragma unroll
    for (int q = 0; q < 19; q++) {
        float cu = 3.f * (c[q][0] * u[0] + c[q][1] * u[1] + c[q][2] * u[2]);
        float feq = w[q] * rho * (1.f + cu + 0.5f * cu * cu - uu);
        dst[q * cells + n] = f[q] + omega * (feq - f[q]);
    }
}
