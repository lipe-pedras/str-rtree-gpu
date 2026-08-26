// bench_context_init.cu
// Measures the one-time CUDA start-up costs that any small-scale GPU-vs-CPU
// comparison must report separately, or it ends up measuring the driver.
//
// PROCEDURE (single process, in order, each timed as a lap):
//   1. cudaFree(0)          — forces CUDA context creation
//   2. cudaEventCreate x2   — event pool setup
//   3. cudaMalloc           — first and second device allocation
//   4. cudaDeviceSynchronize
//   5. first thrust::sort   — CUB module load / kernel JIT for this arch
//   6. second thrust::sort  — the steady-state cost, for contrast
//
// COLD vs WARM: with persistence mode disabled the driver tears down GPU state
// after an idle period, so step 1 costs ~10x more on the first process after
// idling than on a process launched right after another CUDA program.  Run
// this once after the GPU has been idle for a while (cold), then immediately
// again (warm), and compare.  Mixing cold and warm runs inside one benchmark
// table will silently corrupt it.
//
//   Check persistence mode with:  nvidia-smi --query-gpu=persistence_mode --format=csv
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 bench_context_init.cu -o bench_context_init

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main() {
    auto t0 = Clock::now();
    auto lap = [&](const char* what) {
        printf("%-38s %10.2f ms\n", what, Ms(Clock::now() - t0).count());
        t0 = Clock::now();
    };

    cudaFree(0);                                        lap("cudaFree(0)  [context creation]");

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);           lap("cudaEventCreate x2");

    void* p = nullptr; cudaMalloc(&p, 24u << 20);       lap("cudaMalloc 24 MB  [first]");
    void* q = nullptr; cudaMalloc(&q,  4u << 20);       lap("cudaMalloc 4 MB   [second]");
    cudaDeviceSynchronize();                            lap("cudaDeviceSynchronize");

    thrust::sort(thrust::device, thrust::device_pointer_cast((int*)q),
                 thrust::device_pointer_cast(((int*)q) + 1024));
    cudaDeviceSynchronize();                            lap("first thrust::sort  [module load]");

    thrust::sort(thrust::device, thrust::device_pointer_cast((int*)q),
                 thrust::device_pointer_cast(((int*)q) + 1024));
    cudaDeviceSynchronize();                            lap("second thrust::sort [steady state]");

    cudaFree(p); cudaFree(q);
    return 0;
}
