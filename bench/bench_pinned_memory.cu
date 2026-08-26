// bench_pinned_memory.cu
// Measures the cost of pinned (page-locked) host memory against the transfer
// bandwidth it buys — the measurement behind MAX_PINNED_CHUNK_BYTES.
//
// The question: the loader sorts in a pinned buffer, so how large should that
// buffer be?  The intuition is "as large as the GPU can sort", which is wrong
// if allocation cost grows with size while bandwidth does not.
//
// PROCEDURE, per buffer size, repeated `trials` times (best reported):
//   1. time cudaMallocHost(size)
//   2. memset the whole buffer, so every page is resident before transfer
//      (otherwise the first H2D would also be paying first-touch faults)
//   3. time a H2D cudaMemcpy of min(size, 1 GB) and derive GB/s
//   4. time cudaFreeHost(size)
//   5. for reference, time malloc + memset of the same size, to separate
//      "cost of memory" from "cost of PINNING memory"
//
// Context creation is forced before the sweep so it is not charged to the
// first allocation.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 bench_pinned_memory.cu -o bench_pinned_memory
// Run:   ./bench_pinned_memory [trials]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 3;
    cudaFree(0);   // context creation outside the measured region

    printf("# best of %d trials\n", trials);
    printf("%10s %16s %14s %10s %16s %14s\n",
           "size", "cudaMallocHost", "cudaFreeHost", "ms/MB",
           "malloc+memset", "H2D GB/s");

    for (size_t mb : {64ul, 128ul, 256ul, 512ul, 1024ul, 1350ul}) {
        size_t b = mb << 20;
        double best_alloc = 1e18, best_free = 1e18, best_h2d = 1e18, best_pageable = 1e18;

        for (int r = 0; r < trials; r++) {
            void* p = nullptr;
            auto t = Clock::now();
            cudaMallocHost(&p, b);
            best_alloc = std::min(best_alloc, Ms(Clock::now() - t).count());
            if (!p) { fprintf(stderr, "cudaMallocHost(%zu MB) failed\n", mb); return 1; }

            size_t xb = std::min(b, 1ul << 30);
            void* d = nullptr; cudaMalloc(&d, xb);
            memset(p, 1, b);                       // make every page resident

            t = Clock::now();
            cudaMemcpy(d, p, xb, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            double h2d = Ms(Clock::now() - t).count();
            best_h2d = std::min(best_h2d, h2d);
            cudaFree(d);

            t = Clock::now();
            cudaFreeHost(p);
            best_free = std::min(best_free, Ms(Clock::now() - t).count());

            t = Clock::now();
            void* q = malloc(b); memset(q, 1, b);
            best_pageable = std::min(best_pageable, Ms(Clock::now() - t).count());
            free(q);
        }

        size_t xb = std::min(b, 1ul << 30);
        printf("%8zu M %14.1f ms %11.1f ms %10.3f %14.1f ms %14.2f\n",
               mb, best_alloc, best_free, best_alloc / mb,
               best_pageable, (xb / 1e9) / (best_h2d / 1000.0));
    }
    return 0;
}
