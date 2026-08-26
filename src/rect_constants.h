// rect_constants.h
// Configuration for the textbook-layout STR R-Tree bulk-loader over rectangles.
//
// Run ./bin/gpu_info to see your GPU specs.
// Everything here is denominated in BYTES or in PAGES — never in abstract
// "fanout" numbers.  The I/O block is the natural unit for an out-of-core
// structure, so entry capacities are derived from it, not the other way round.

#ifndef RECT_CONSTANTS_H
#define RECT_CONSTANTS_H

#include <cstddef>
#include <cstdint>

// =========================================================
// Page geometry
// =========================================================

// One node == one page.  Nodes are page-ALIGNED in the file (page 0 holds the
// file header), so expanding a node is exactly one page fault — never two.
constexpr size_t PAGE_BYTES = 4096;

// =========================================================
// GPU memory budget
// =========================================================

// ~80% of VRAM, leaving room for the CUDA runtime and CUB scratch.
//   4 GB GPU -> 3.2 GB -> 3435973836ULL
//   8 GB GPU -> 6.4 GB -> 6871947674ULL
//  12 GB GPU -> 9.6 GB -> 10307921511ULL
//  24 GB GPU -> 19.2 GB -> 20615843020ULL
constexpr size_t USABLE_GPU_BYTES = 3161037209ULL;   // 2.94 GB (RTX 3050 4GB)

// =========================================================
// Host RAM budget
// =========================================================

// Budget for caching sorted runs in RAM instead of spilling them to disk.
// When the whole dataset fits here, the build costs exactly one read of the
// input and one write of the output — the disk is touched a single time.
// Set to ~40-60% of physical RAM (the merge and page buffers need headroom).
constexpr size_t USABLE_RAM_BYTES = 5368709120ULL;   // 5 GB

// Host-side budget for the leaf-building phase.  It has to hold two slice
// buffers (entries) AND two page buffers (packed nodes) for the
// sort-while-writing pipeline, so it is capped separately from the GPU budget.
constexpr size_t PHASE_LEAF_HOST_BYTES = 2ULL * 1024 * 1024 * 1024;  // 2 GB

// Budget shared by all k-way merge read buffers.
constexpr size_t MERGE_BUFFER_BYTES = 2ULL * 1024 * 1024 * 1024;     // 2 GB

// Upper bound on a single pinned staging buffer.
//
// MEASURED: cudaMallocHost costs ~1 ms per MB and cudaFreeHost ~0.25 ms per MB,
// both LINEAR in size, while H2D bandwidth is FLAT at ~13.3 GB/s from 64 MB
// upward (64 MB: 13.07 GB/s, 1350 MB: 13.34 GB/s).  A giant pinned buffer
// therefore costs seconds to allocate and buys nothing in transfer rate:
// 2 x 1350 MB of staging cost 2.7 s to allocate plus 0.7 s to free.
//
// So the GPU sort chunk is capped here rather than at the GPU budget.  The
// cost of a smaller chunk is more runs (k) for the merge, which raises the
// heap merge's O(N log k) comparison count -- see the sweep in context.md.
constexpr size_t MAX_PINNED_CHUNK_BYTES = 268435456ULL;   // 256 MB

// Buffered page-append size for the (small) internal levels.
constexpr size_t PAGE_WRITE_BUFFER_BYTES = 64ULL * 1024 * 1024;      // 64 MB

// =========================================================
// Fill factors (defaults; overridable on the command line)
// =========================================================

// Fraction of each node actually used, so a bulk-loaded tree can be left with
// free space for later inserts.  1.0 == fully packed.
constexpr float DEFAULT_FILL_LEAF     = 1.0f;
constexpr float DEFAULT_FILL_INTERNAL = 1.0f;

// =========================================================
// Sort dispatch
// =========================================================

// Below this many entries a level is sorted on the CPU: the PCIe round trip
// costs more than the sort saves.
//
// MEASURED on RTX 3050 Laptop / Ryzen 5 6600H, pinned buffers, 24-byte Entry
// (GPU column includes H2D + key build + radix sort_by_key + D2H):
//
//        n     std::sort    GPU total    GPU sort only   winner
//     1 024      0.010 ms     0.185 ms        0.165 ms   CPU
//     4 096      0.191 ms     0.220 ms        0.170 ms   CPU
//    16 384      1.155 ms     0.262 ms        0.184 ms   GPU   4.4x
//   262 144     24.245 ms     1.576 ms        0.611 ms   GPU  15.4x
// 1 048 576    100.365 ms     5.664 ms        1.888 ms   GPU  17.7x
// 16 777 216  1814.560 ms    86.267 ms       25.978 ms   GPU  21.0x
//
// Two things worth reading off that table:
//   * The crossover is around 8 Ki entries — three orders of magnitude lower
//     than the intuitive guess.  Once the data is already in a pinned buffer,
//     even a tiny sort is worth shipping to the GPU.
//   * At the top end the kernel is ~70x faster than std::sort, but PCIe eats
//     ~70% of the win (60 of the 86 ms are transfer, ~13.4 GB/s on PCIe 4.0
//     x8), leaving 21x end-to-end.  The transfer tax, not the sort, is what
//     bounds this design.
constexpr size_t GPU_SORT_MIN_ELEMS = 8192;

#endif // RECT_CONSTANTS_H
