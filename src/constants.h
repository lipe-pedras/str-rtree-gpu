// constants.h
// GPU and algorithm configuration for external STR
//
// Run ./gpu_info to see your GPU specs and suggested values.
// Adjust the values below to match your hardware.

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstddef>

// =========================================================
// GPU Memory Budget
// =========================================================

// Total GPU memory you want the program to use.
// Should be ~80% of your GPU's total VRAM to leave room for
// CUDA runtime overhead, Thrust scratch space, etc.
//
// Examples:
//   2 GB GPU  ->  1.6 GB  ->  1717986918ULL
//   4 GB GPU  ->  3.2 GB  ->  3435973836ULL
//   6 GB GPU  ->  4.8 GB  ->  5153960755ULL
//   8 GB GPU  ->  6.4 GB  ->  6871947674ULL
//  12 GB GPU  ->  9.6 GB  -> 10307921511ULL
//  16 GB GPU  -> 12.8 GB  -> 13743895347ULL
//  24 GB GPU  -> 19.2 GB  -> 20615843020ULL
constexpr size_t USABLE_GPU_BYTES = 3161037209ULL;

// =========================================================
// Host RAM Budget
// =========================================================

// If the entire dataset fits in this much RAM, the program
// keeps everything in memory — only 1 read + 1 write to disk.
// Otherwise it falls back to the external (disk-based) path.
//
// Set to ~70-80% of your physical RAM.
// The program needs the data buffer + some extra for the
// in-memory merge (~50% overhead from std::inplace_merge).
//
// Examples:
//   8 GB RAM  ->  use  5 GB  ->   5368709120ULL
//  16 GB RAM  ->  use 11 GB  ->  11811160064ULL
//  32 GB RAM  ->  use 24 GB  ->  25769803776ULL
//  64 GB RAM  ->  use 48 GB  ->  51539607552ULL
constexpr size_t USABLE_RAM_BYTES = 1ULL;  // 11 GB

// =========================================================
// Chunk Sizes (derived from GPU budget)
// =========================================================

// Points per chunk for gpu_sort().
// Thrust sort needs ~2x the data size in GPU memory (input + internal scratch).
// So we use USABLE_GPU_BYTES / 2 / sizeof(Point).
constexpr size_t SORT_CHUNK_POINTS = USABLE_GPU_BYTES / 2 / 8;  // sizeof(Point) == 8

// =========================================================
// R-Tree / STR Parameters
// =========================================================

// Maximum entries per R-Tree node (leaf or internal).
// Leaf nodes hold up to this many points.
// Internal nodes hold up to this many child MBRs.
// Typical values: 50–256.  Larger → shallower tree, larger nodes.
constexpr size_t RTREE_NODE_CAPACITY = 128;

// Maximum points per vertical slice for STR Y-sorting.
// Capped at SORT_CHUNK_POINTS (must fit in one GPU sort call).
// The actual slice size is computed at runtime based on
// RTREE_NODE_CAPACITY and the dataset size (see compute_str_slice_size).
constexpr size_t STR_MAX_SLICE = SORT_CHUNK_POINTS;

#endif // CONSTANTS_H
