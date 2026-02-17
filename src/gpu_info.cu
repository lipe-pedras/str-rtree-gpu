// gpu_info.cu
// Displays GPU specifications to help configure constants.h
// Compile: nvcc -O3 gpu_info.cu -o gpu_info

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return 1;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        double total_mb = prop.totalGlobalMem / (1024.0 * 1024.0);
        double total_gb = total_mb / 1024.0;

        // A safe usable amount is ~80% of total VRAM
        size_t safe_bytes = (size_t)(prop.totalGlobalMem * 0.80);
        double safe_mb    = safe_bytes / (1024.0 * 1024.0);
        double safe_gb    = safe_mb / 1024.0;

        // For sort we need 1 buffer + Thrust scratch (~2x),
        // so usable is ~1/2 of safe memory
        size_t sort_chunk_bytes  = safe_bytes / 2;
        size_t sort_chunk_points = sort_chunk_bytes / 8;

        std::cout << "========================================\n"
                  << "GPU " << dev << ": " << prop.name << "\n"
                  << "========================================\n"
                  << std::fixed << std::setprecision(2)
                  << "  Total VRAM:           " << total_mb << " MB ("
                  << total_gb << " GB)\n"
                  << "  Safe usable (~80%):   " << safe_mb << " MB ("
                  << safe_gb << " GB)\n"
                  << "  Compute capability:   " << prop.major << "."
                  << prop.minor << "\n"
                  << "  SM count:             " << prop.multiProcessorCount << "\n"
                  << "  Max threads/block:    " << prop.maxThreadsPerBlock << "\n"
                  << "  Warp size:            " << prop.warpSize << "\n"
                  << "  Memory bus width:     " << prop.memoryBusWidth << " bits\n"
                  << "  Memory clock rate:    " << prop.memoryClockRate / 1000
                  << " MHz\n"
                  << "  L2 cache size:        " << prop.l2CacheSize / 1024
                  << " KB\n"
                  << "\n"
                  << "--- Suggested constants.h values ---\n"
                  << "  USABLE_GPU_BYTES:     " << safe_bytes
                  << "  (" << safe_gb << " GB)\n"
                  << "  SORT_CHUNK_POINTS:    " << sort_chunk_points
                  << "  (" << sort_chunk_points * 8.0 / (1024*1024) << " MB)\n"
                  << std::endl;
    }

    // Also show current free/total at runtime
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "--- Runtime memory status ---\n"
              << "  Free:  " << free_bytes / (1024.0*1024.0) << " MB\n"
              << "  Total: " << total_bytes / (1024.0*1024.0) << " MB\n"
              << "  Used:  " << (total_bytes - free_bytes) / (1024.0*1024.0)
              << " MB (by other processes)\n" << std::endl;

    return 0;
}
