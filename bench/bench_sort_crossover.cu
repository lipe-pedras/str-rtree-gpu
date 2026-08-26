// bench_sort_crossover.cu
// Measures where GPU sorting starts to beat CPU sorting for the 24-byte Entry
// records used by the STR bulk-loader — i.e. the break-even point that decides
// GPU_SORT_MIN_ELEMS in rect_constants.h.
//
// PROCEDURE
//   For each n in a geometric sweep (x4 per step):
//     CPU arm: refill a pinned host buffer from an untouched master copy, then
//              time std::sort with the same centroid comparator the loader uses.
//     GPU arm: refill the same buffer, then time the FULL round trip —
//              H2D copy, key-extraction kernel, radix sort_by_key, D2H copy.
//   Both arms are timed with the refill OUTSIDE the timer, so neither is
//   charged for staging its own input.  The best of `trials` runs is reported
//   for each arm (best-of favours both arms equally and suppresses scheduler
//   noise on a laptop).
//
// DELIBERATELY EXCLUDED, because production code pays them once, not per sort:
//   - CUDA context creation (forced before the sweep begins)
//   - cudaMalloc of the device buffers (hoisted out of the loop)
//   - cudaMallocHost of the pinned buffer (hoisted; measured separately by
//     bench_pinned_memory, and it must be ADDED BACK when reasoning about a
//     one-shot sort rather than a reused buffer)
//
// The `gpu_sort_only` column isolates the radix sort itself, so the difference
// against `gpu_total` is the PCIe transfer tax.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -I../src bench_sort_crossover.cu -o bench_sort_crossover
// Run:   ./bench_sort_crossover [max_log2_n] [trials]

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "rect_rtree_format.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

__host__ __device__ inline uint32_t order_float(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void k_make_keys(const Entry* s, uint32_t* k, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    k[i] = order_float(s[i].mbr.min_x + s[i].mbr.max_x);
}

int main(int argc, char** argv) {
    int max_log2 = argc > 1 ? atoi(argv[1]) : 24;
    int trials   = argc > 2 ? atoi(argv[2]) : 5;
    const size_t MAXN = 1ull << max_log2;

    cudaFree(0);   // force context creation OUTSIDE the measured region

    Entry*    h  = nullptr; cudaMallocHost(&h,  MAXN * sizeof(Entry));
    Entry*    d  = nullptr; cudaMalloc(&d,      MAXN * sizeof(Entry));
    uint32_t* dk = nullptr; cudaMalloc(&dk,     MAXN * sizeof(uint32_t));
    if (!h || !d || !dk) { fprintf(stderr, "allocation failed\n"); return 1; }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0, 1000);
    std::vector<Entry> master(MAXN);
    for (size_t i = 0; i < MAXN; i++) {
        float x = u(rng), y = u(rng);
        master[i] = Entry{{x, y, x + 1, y + 1}, i};
    }

    printf("# best of %d trials, 24-byte Entry, pinned host buffer\n", trials);
    printf("%12s %12s %12s %14s %10s %10s\n",
           "n", "cpu_ms", "gpu_total", "gpu_sort_only", "speedup", "winner");

    for (size_t n = 1024; n <= MAXN; n *= 4) {
        double cpu = 1e18, gpu = 1e18, gsort = 0;

        for (int r = 0; r < trials; r++) {
            memcpy(h, master.data(), n * sizeof(Entry));
            auto a = Clock::now();
            std::sort(h, h + n, [](const Entry& p, const Entry& q) {
                return centroid_key_x(p.mbr) < centroid_key_x(q.mbr); });
            cpu = std::min(cpu, Ms(Clock::now() - a).count());
        }

        for (int r = 0; r < trials; r++) {
            memcpy(h, master.data(), n * sizeof(Entry));
            auto a = Clock::now();
            cudaMemcpy(d, h, n * sizeof(Entry), cudaMemcpyHostToDevice);
            k_make_keys<<<(unsigned)((n + 255) / 256), 256>>>(d, dk, n);
            auto b = Clock::now();
            thrust::sort_by_key(thrust::device,
                                thrust::device_pointer_cast(dk),
                                thrust::device_pointer_cast(dk + n),
                                thrust::device_pointer_cast(d));
            cudaDeviceSynchronize();
            double sm = Ms(Clock::now() - b).count();
            cudaMemcpy(h, d, n * sizeof(Entry), cudaMemcpyDeviceToHost);
            double tot = Ms(Clock::now() - a).count();
            if (tot < gpu) { gpu = tot; gsort = sm; }
        }

        printf("%12zu %12.3f %12.3f %14.3f %9.1fx %10s\n",
               n, cpu, gpu, gsort, cpu / gpu, cpu < gpu ? "CPU" : "GPU");
    }

    cudaFreeHost(h); cudaFree(d); cudaFree(dk);
    return 0;
}
