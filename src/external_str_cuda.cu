// external_str_cuda.cu
// STR (Sort-Tile-Recursive) R-Tree bulk-loading with GPU pipeline
// Supports datasets larger than VRAM via triple-buffered external sorting.
//
// Output file format:
//   [RTreeHeader  – 32 bytes]
//   [RTreeNode[]  – num_nodes × 32 bytes]
//   [Point[]      – num_points × 8 bytes]
//
// Root node is always nodes[num_nodes - 1].
// Leaf nodes reference contiguous ranges of points.
// Internal nodes reference contiguous ranges of child nodes.
//
// Compile: nvcc -O3 -std=c++17 external_str_cuda.cu -o external_str -lpthread

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <chrono>
#include <future>
#include <algorithm>
#include <queue>

#include "constants.h"

// =========================================================
// Timing infrastructure
// =========================================================

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct TimingStats {
    double disk_read_ms   = 0;
    double disk_write_ms  = 0;
    double h2d_ms         = 0;
    double d2h_ms         = 0;
    double gpu_compute_ms = 0;
    double gpu_alloc_ms   = 0;

    size_t bytes_read     = 0;
    size_t bytes_written  = 0;
    size_t bytes_h2d      = 0;
    size_t bytes_d2h      = 0;

    void print(const char* phase) const {
        auto bw = [](size_t bytes, double ms) -> double {
            return (ms > 0) ? (bytes / (1024.0*1024.0)) / (ms / 1000.0) : 0;
        };
        std::cout << "\n===== Timing: " << phase << " =====\n"
                  << std::fixed << std::setprecision(2)
                  << "  Disk read:     " << disk_read_ms   << " ms  ("
                  << bw(bytes_read, disk_read_ms) << " MB/s, "
                  << bytes_read / (1024.0*1024.0) << " MB)\n"
                  << "  Disk write:    " << disk_write_ms  << " ms  ("
                  << bw(bytes_written, disk_write_ms) << " MB/s, "
                  << bytes_written / (1024.0*1024.0) << " MB)\n"
                  << "  GPU alloc:     " << gpu_alloc_ms   << " ms\n"
                  << "  H2D transfer:  " << h2d_ms         << " ms  ("
                  << bw(bytes_h2d, h2d_ms) << " MB/s, "
                  << bytes_h2d / (1024.0*1024.0) << " MB)\n"
                  << "  GPU compute:   " << gpu_compute_ms << " ms\n"
                  << "  D2H transfer:  " << d2h_ms         << " ms  ("
                  << bw(bytes_d2h, d2h_ms) << " MB/s, "
                  << bytes_d2h / (1024.0*1024.0) << " MB)\n"
                  << "  TOTAL:         "
                  << (disk_read_ms + disk_write_ms + gpu_alloc_ms
                      + h2d_ms + gpu_compute_ms + d2h_ms) << " ms\n";
    }

    TimingStats& operator+=(const TimingStats& o) {
        disk_read_ms   += o.disk_read_ms;
        disk_write_ms  += o.disk_write_ms;
        h2d_ms         += o.h2d_ms;
        d2h_ms         += o.d2h_ms;
        gpu_compute_ms += o.gpu_compute_ms;
        gpu_alloc_ms   += o.gpu_alloc_ms;
        bytes_read     += o.bytes_read;
        bytes_written  += o.bytes_written;
        bytes_h2d      += o.bytes_h2d;
        bytes_d2h      += o.bytes_d2h;
        return *this;
    }
};

TimingStats g_stats;  // global accumulator

// =========================================================
// Data structure
// =========================================================

struct Point {
    float x;
    float y;
};

static_assert(sizeof(Point) == 8);
static_assert(RTREE_NODE_BYTES / sizeof(Point) == RTREE_LEAF_CAPACITY,
              "RTREE_LEAF_CAPACITY out of sync with Point size");

// =========================================================
// R-Tree structures
// =========================================================

struct MBR {
    float min_x, min_y, max_x, max_y;
};

struct RTreeNode {
    MBR      mbr;           // 16 bytes
    uint64_t first_child;   // leaf: index into points array
                            // internal: index into nodes array
    uint32_t num_children;  // 4 bytes
    uint32_t is_leaf;       // 4 bytes   (1 = leaf, 0 = internal)
};

static_assert(sizeof(RTreeNode) == 32);
static_assert(RTREE_NODE_BYTES / sizeof(RTreeNode) == RTREE_INTERNAL_CAPACITY,
              "RTREE_INTERNAL_CAPACITY out of sync with RTreeNode size");

struct RTreeHeader {
    uint32_t magic;              // 0x52545245  "RTRE"
    uint32_t leaf_capacity;      // points per leaf node
    uint32_t height;
    uint32_t internal_capacity;  // children per internal node
    uint64_t num_points;
    uint64_t num_nodes;
};

static_assert(sizeof(RTreeHeader) == 32);

// =========================================================
// Comparators
// =========================================================

struct CompareX {
    __host__ __device__
    bool operator()(const Point& a, const Point& b) const {
        return a.x < b.x;
    }
};

struct CompareY {
    __host__ __device__
    bool operator()(const Point& a, const Point& b) const {
        return a.y < b.y;
    }
};

// =========================================================
// GPU SORT (chunk local)  — with detailed timing
// =========================================================

void gpu_sort(Point* host_data, size_t n, bool by_x) {
    size_t bytes = n * sizeof(Point);
    Point* d_ptr_raw = nullptr;

    // --- GPU alloc ---
    auto t0 = Clock::now();
    cudaMalloc(&d_ptr_raw, bytes);
    cudaDeviceSynchronize();
    auto t1 = Clock::now();
    g_stats.gpu_alloc_ms += Ms(t1 - t0).count();

    // --- H2D ---
    cudaMemcpy(d_ptr_raw, host_data, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t2 = Clock::now();
    g_stats.h2d_ms    += Ms(t2 - t1).count();
    g_stats.bytes_h2d += bytes;

    // --- GPU sort ---
    thrust::device_ptr<Point> d(d_ptr_raw);
    if (by_x)
        thrust::sort(d, d + n, CompareX());
    else
        thrust::sort(d, d + n, CompareY());
    cudaDeviceSynchronize();
    auto t3 = Clock::now();
    g_stats.gpu_compute_ms += Ms(t3 - t2).count();

    // --- D2H ---
    cudaMemcpy(host_data, d_ptr_raw, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto t4 = Clock::now();
    g_stats.d2h_ms    += Ms(t4 - t3).count();
    g_stats.bytes_d2h += bytes;

    cudaFree(d_ptr_raw);
}

// =========================================================
// GPU SORT with pre-allocated device buffer (no alloc/free)
// =========================================================

void gpu_sort_reuse(Point* host_data, size_t n, bool by_x, Point* d_ptr_raw) {
    size_t bytes = n * sizeof(Point);

    // --- H2D ---
    auto t0 = Clock::now();
    cudaMemcpy(d_ptr_raw, host_data, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t1 = Clock::now();
    g_stats.h2d_ms    += Ms(t1 - t0).count();
    g_stats.bytes_h2d += bytes;

    // --- GPU sort ---
    thrust::device_ptr<Point> d(d_ptr_raw);
    if (by_x)
        thrust::sort(d, d + n, CompareX());
    else
        thrust::sort(d, d + n, CompareY());
    cudaDeviceSynchronize();
    auto t2 = Clock::now();
    g_stats.gpu_compute_ms += Ms(t2 - t1).count();

    // --- D2H ---
    cudaMemcpy(host_data, d_ptr_raw, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto t3 = Clock::now();
    g_stats.d2h_ms    += Ms(t3 - t2).count();
    g_stats.bytes_d2h += bytes;
}

// =========================================================
// Generate initial sorted runs (GPU) — triple-buffered pipeline
//
// Overlaps disk I/O with GPU sort using 3 pinned host buffers:
//   buf[i] -> GPU sort  (main thread)
//   buf[j] <- disk read (async reader)
//   buf[k] -> disk write (async writer)
// =========================================================

size_t generate_runs(const std::string& input,
                     size_t points_per_chunk,
                     bool by_x,
                     const std::string& prefix) {

    size_t chunk_bytes = points_per_chunk * sizeof(Point);

    // Pinned memory for faster H2D/D2H and safe async access
    Point* bufs[3];
    for (int i = 0; i < 3; i++)
        cudaMallocHost(&bufs[i], chunk_bytes);

    std::ifstream in(input, std::ios::binary);
    size_t run_id = 0;

    // --- Pre-read first chunk synchronously ---
    auto t0 = Clock::now();
    in.read(reinterpret_cast<char*>(bufs[0]), chunk_bytes);
    size_t n0 = in.gcount() / sizeof(Point);
    auto t1 = Clock::now();
    g_stats.disk_read_ms += Ms(t1 - t0).count();
    g_stats.bytes_read   += n0 * sizeof(Point);

    if (n0 == 0) {
        for (int i = 0; i < 3; i++) cudaFreeHost(bufs[i]);
        in.close();
        return 0;
    }

    int    sort_idx     = 0;
    size_t sort_n       = n0;
    int    read_idx     = 1;
    int    write_idx    = -1;      // nothing to write yet
    size_t write_n      = 0;
    size_t write_run_id = 0;
    bool   input_done   = false;

    while (sort_n > 0) {
        // 1) Async READ of next chunk into bufs[read_idx]
        std::future<std::pair<size_t, double>> read_fut;
        bool do_read = !input_done;
        if (do_read) {
            Point* rp = bufs[read_idx];
            read_fut = std::async(std::launch::async,
                [&in, rp, chunk_bytes]() -> std::pair<size_t, double> {
                    auto t0 = Clock::now();
                    in.read(reinterpret_cast<char*>(rp), chunk_bytes);
                    size_t n = in.gcount() / sizeof(Point);
                    auto t1 = Clock::now();
                    return {n, Ms(t1 - t0).count()};
                });
        }

        // 2) Async WRITE of previously sorted chunk from bufs[write_idx]
        std::future<double> write_fut;
        bool do_write = (write_idx >= 0);
        if (do_write) {
            Point* wp = bufs[write_idx];
            size_t wn = write_n;
            size_t wrid = write_run_id;
            write_fut = std::async(std::launch::async,
                [wp, wn, &prefix, wrid]() -> double {
                    auto t0 = Clock::now();
                    std::ofstream f(prefix + std::to_string(wrid) + ".bin",
                                    std::ios::binary);
                    f.write(reinterpret_cast<char*>(wp),
                            wn * sizeof(Point));
                    f.close();
                    auto t1 = Clock::now();
                    return Ms(t1 - t0).count();
                });
        }

        // 3) GPU sort current chunk (main thread, overlaps with I/O)
        gpu_sort(bufs[sort_idx], sort_n, by_x);
        size_t cur_run_id = run_id++;

        // 4) Collect async results
        size_t next_n = 0;
        if (do_read) {
            std::pair<size_t, double> res = read_fut.get();
            g_stats.disk_read_ms += res.second;
            g_stats.bytes_read   += res.first * sizeof(Point);
            next_n = res.first;
            if (res.first == 0) input_done = true;
        }
        if (do_write) {
            double ms = write_fut.get();
            g_stats.disk_write_ms += ms;
            g_stats.bytes_written += write_n * sizeof(Point);
        }

        // 5) Rotate: sorted chunk becomes pending write,
        //           read chunk becomes next to sort
        write_idx    = sort_idx;
        write_n      = sort_n;
        write_run_id = cur_run_id;

        sort_n = next_n;
        if (sort_n > 0) {
            sort_idx = read_idx;
            // Free buffer = the one that’s neither sort nor write
            for (int i = 0; i < 3; i++) {
                if (i != sort_idx && i != write_idx) {
                    read_idx = i;
                    break;
                }
            }
        }
    }

    // Flush: write the last sorted chunk
    if (write_idx >= 0) {
        auto tw0 = Clock::now();
        std::ofstream f(prefix + std::to_string(write_run_id) + ".bin",
                        std::ios::binary);
        f.write(reinterpret_cast<char*>(bufs[write_idx]),
                write_n * sizeof(Point));
        f.close();
        auto tw1 = Clock::now();
        g_stats.disk_write_ms += Ms(tw1 - tw0).count();
        g_stats.bytes_written += write_n * sizeof(Point);
    }

    in.close();
    for (int i = 0; i < 3; i++) cudaFreeHost(bufs[i]);
    return run_id;
}

// =========================================================
// K-way merge: merges all run files in a single pass.
// Uses buffered readers + min-heap.  Deletes inputs after use.
// =========================================================

void kway_merge_files(size_t run_count,
                      const std::string& prefix,
                      const std::string& output_path,
                      bool by_x) {
    if (run_count == 0) return;
    if (run_count == 1) {
        std::filesystem::rename(prefix + "0.bin", output_path);
        return;
    }

    // Buffer size per stream: balance RAM usage vs I/O throughput.
    // Total merge memory ≈ (run_count + 1) × buf_points × sizeof(Point).
    constexpr size_t MAX_MERGE_MEM  = 2ULL * 1024 * 1024 * 1024; // 2 GB
    constexpr size_t MIN_BUF_POINTS = 4096;
    constexpr size_t MAX_BUF_POINTS = 16 * 1024 * 1024;          // 128 MB

    size_t buf_points = MAX_MERGE_MEM / ((run_count + 1) * sizeof(Point));
    buf_points = std::max(buf_points, MIN_BUF_POINTS);
    buf_points = std::min(buf_points, MAX_BUF_POINTS);

    std::cout << "  K-way merge: " << run_count << " runs, "
              << buf_points << " pts/buf ("
              << buf_points * sizeof(Point) / (1024.0*1024.0) << " MB)\n";

    // --- Buffered reader per run ---
    struct RunReader {
        std::ifstream file;
        std::vector<Point> buf;
        size_t pos = 0, end = 0;
        bool eof = false;
        std::string path;

        void open(const std::string& p, size_t buf_sz) {
            path = p;
            buf.resize(buf_sz);
            file.open(p, std::ios::binary);
            refill();
        }

        void refill() {
            if (eof) return;
            auto t0 = Clock::now();
            file.read(reinterpret_cast<char*>(buf.data()),
                      buf.size() * sizeof(Point));
            end = file.gcount() / sizeof(Point);
            pos = 0;
            auto t1 = Clock::now();
            g_stats.disk_read_ms += Ms(t1 - t0).count();
            g_stats.bytes_read   += end * sizeof(Point);
            if (end == 0) eof = true;
        }

        const Point& current() const { return buf[pos]; }

        bool advance() {
            pos++;
            if (pos >= end) {
                refill();
                if (eof) return false;
            }
            return true;
        }

        void close_and_delete() {
            file.close();
            std::filesystem::remove(path);
        }
    };

    std::vector<RunReader> readers(run_count);
    for (size_t i = 0; i < run_count; i++)
        readers[i].open(prefix + std::to_string(i) + ".bin", buf_points);

    // --- Min-heap of (point, run_index) ---
    struct HeapEntry {
        Point  pt;
        size_t run;
    };

    auto cmp = [by_x](const HeapEntry& a, const HeapEntry& b) {
        // Inverted: std::priority_queue is a max-heap, so "greater" → min-heap
        return by_x ? (a.pt.x > b.pt.x) : (a.pt.y > b.pt.y);
    };

    std::vector<HeapEntry> heap;
    heap.reserve(run_count);

    // Seed heap with one entry per run
    for (size_t i = 0; i < run_count; i++) {
        if (!readers[i].eof)
            heap.push_back({readers[i].current(), i});
    }
    std::make_heap(heap.begin(), heap.end(), cmp);

    // --- Output buffer ---
    std::vector<Point> out_buf(buf_points);
    size_t out_pos = 0;

    std::ofstream fout(output_path, std::ios::binary);

    auto flush_out = [&]() {
        if (out_pos == 0) return;
        auto t0 = Clock::now();
        fout.write(reinterpret_cast<char*>(out_buf.data()),
                   out_pos * sizeof(Point));
        auto t1 = Clock::now();
        g_stats.disk_write_ms += Ms(t1 - t0).count();
        g_stats.bytes_written += out_pos * sizeof(Point);
        out_pos = 0;
    };

    // --- Merge loop ---
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        HeapEntry top = heap.back();
        heap.pop_back();

        out_buf[out_pos++] = top.pt;
        if (out_pos >= buf_points) flush_out();

        RunReader& r = readers[top.run];
        if (r.advance()) {
            heap.push_back({r.current(), top.run});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else {
            r.close_and_delete();
        }
    }

    flush_out();
    fout.close();
}

// =========================================================
// STR slice size computation
// =========================================================

size_t compute_str_slice_size(size_t total_points) {
    size_t leaf_cap = RTREE_LEAF_CAPACITY;
    size_t int_cap  = RTREE_INTERNAL_CAPACITY;
    size_t num_leaves = (total_points + leaf_cap - 1) / leaf_cap;
    size_t num_slices = (size_t)std::ceil(std::sqrt((double)num_leaves));

    // Leaves per slice, rounded UP to a multiple of `int_cap` so that
    // level-1 internal node boundaries coincide with slice boundaries.
    // Without this, an internal node can straddle two slices whose
    // Y-sort domains are in different X-ranges, producing spatially
    // disjoint MBRs among its children.
    size_t leaves_per_slice = (num_leaves + num_slices - 1) / num_slices;
    leaves_per_slice = ((leaves_per_slice + int_cap - 1) / int_cap) * int_cap;

    size_t slice_points = leaves_per_slice * leaf_cap;

    // Cap at GPU sort capacity, keeping leaf_cap × int_cap alignment.
    size_t align = leaf_cap * int_cap;
    size_t max_slice = (STR_MAX_SLICE / align) * align;
    if (max_slice < align) {
        std::cout << "Warning: STR_MAX_SLICE too small for leaf/internal node alignment\n";
        max_slice = align;
    }
    return std::min(slice_points, max_slice);
}

// =========================================================
// Pre-compute total number of R-tree nodes.
// No data access needed — purely structural computation.
// =========================================================

size_t precompute_num_nodes(size_t total_points,
                           size_t leaf_cap,
                           size_t internal_cap) {
    size_t level = (total_points + leaf_cap - 1) / leaf_cap;  // leaves
    size_t total = level;
    while (level > 1) {
        level = (level + internal_cap - 1) / internal_cap;
        total += level;
    }
    return total;
}

// =========================================================
// Build internal R-tree levels from a vector of leaf nodes.
// Appends internal nodes to `nodes`. Sets height_out.
// =========================================================

void build_internal_levels(std::vector<RTreeNode>& nodes,
                           size_t num_leaves,
                           size_t internal_cap,
                           uint32_t& height_out) {
    height_out = 1;
    size_t level_start = 0;
    size_t level_count = num_leaves;

    while (level_count > 1) {
        size_t next_count = (level_count + internal_cap - 1) / internal_cap;

        for (size_t i = 0; i < next_count; i++) {
            size_t s   = level_start + i * internal_cap;
            size_t cnt = std::min(internal_cap, level_count - i * internal_cap);

            MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
            for (size_t j = 0; j < cnt; j++) {
                const MBR& c = nodes[s + j].mbr;
                m.min_x = std::min(m.min_x, c.min_x);
                m.min_y = std::min(m.min_y, c.min_y);
                m.max_x = std::max(m.max_x, c.max_x);
                m.max_y = std::max(m.max_y, c.max_y);
            }

            RTreeNode nd{};
            nd.mbr          = m;
            nd.first_child  = s;
            nd.num_children = (uint32_t)cnt;
            nd.is_leaf      = 0;
            nodes.push_back(nd);
        }

        level_start += level_count;
        level_count  = next_count;
        height_out++;
    }
}

// =========================================================
// R-Tree construction (bottom-up from STR-sorted points)
//
// Points must already be in STR order (sorted by X globally,
// then by Y within each vertical slice).  Consecutive groups
// of leaf_cap points form leaf nodes.  Internal levels are
// built by grouping child nodes in batches of internal_cap.
// =========================================================

// --- Build from in-memory point array ---
std::vector<RTreeNode> build_rtree(const Point* points,
                                   size_t num_points,
                                   size_t leaf_cap,
                                   size_t internal_cap,
                                   uint32_t& height_out) {
    std::vector<RTreeNode> nodes;
    size_t num_leaves = (num_points + leaf_cap - 1) / leaf_cap;
    nodes.reserve(num_leaves * 2);  // rough upper bound

    // Leaf level
    for (size_t i = 0; i < num_leaves; i++) {
        size_t start = i * leaf_cap;
        size_t cnt   = std::min(leaf_cap, num_points - start);

        MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
        for (size_t j = 0; j < cnt; j++) {
            const Point& p = points[start + j];
            m.min_x = std::min(m.min_x, p.x);
            m.min_y = std::min(m.min_y, p.y);
            m.max_x = std::max(m.max_x, p.x);
            m.max_y = std::max(m.max_y, p.y);
        }

        RTreeNode nd{};
        nd.mbr          = m;
        nd.first_child  = start;
        nd.num_children = (uint32_t)cnt;
        nd.is_leaf      = 1;
        nodes.push_back(nd);
    }

    // Internal levels (bottom-up)
    height_out = 1;
    size_t level_start = 0;
    size_t level_count = num_leaves;

    while (level_count > 1) {
        size_t next_count = (level_count + internal_cap - 1) / internal_cap;

        for (size_t i = 0; i < next_count; i++) {
            size_t s   = level_start + i * internal_cap;
            size_t cnt = std::min(internal_cap, level_count - i * internal_cap);

            MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
            for (size_t j = 0; j < cnt; j++) {
                const MBR& c = nodes[s + j].mbr;
                m.min_x = std::min(m.min_x, c.min_x);
                m.min_y = std::min(m.min_y, c.min_y);
                m.max_x = std::max(m.max_x, c.max_x);
                m.max_y = std::max(m.max_y, c.max_y);
            }

            RTreeNode nd{};
            nd.mbr          = m;
            nd.first_child  = s;
            nd.num_children = (uint32_t)cnt;
            nd.is_leaf      = 0;
            nodes.push_back(nd);
        }

        level_start += level_count;
        level_count  = next_count;
        height_out++;
    }

    return nodes;
}

// =========================================================
// Write R-Tree to disk
//
// File layout:
//   [RTreeHeader  – 32 bytes]
//   [RTreeNode[]  – num_nodes * 32 bytes]
//   [Point[]      – num_points * 8 bytes]
//
// Root node is always nodes[num_nodes - 1].
// Leaf nodes reference points by index into the Point array.
// Internal nodes reference children by index into the Node array.
// =========================================================

void write_rtree_file(const std::string& path,
                      const std::vector<RTreeNode>& nodes,
                      uint32_t height,
                      const Point* points,
                      size_t num_points) {
    RTreeHeader hdr{};
    hdr.magic             = 0x52545245;  // "RTRE"
    hdr.leaf_capacity     = (uint32_t)RTREE_LEAF_CAPACITY;
    hdr.height            = height;
    hdr.internal_capacity = (uint32_t)RTREE_INTERNAL_CAPACITY;
    hdr.num_points        = num_points;
    hdr.num_nodes         = nodes.size();

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char*>(nodes.data()),
              nodes.size() * sizeof(RTreeNode));
    out.write(reinterpret_cast<const char*>(points),
              num_points * sizeof(Point));
    out.close();

    std::cout << "  R-Tree written: " << nodes.size() << " nodes, "
              << height << " levels, root MBR = ["
              << nodes.back().mbr.min_x << ", "
              << nodes.back().mbr.min_y << "] x ["
              << nodes.back().mbr.max_x << ", "
              << nodes.back().mbr.max_y << "]\n";
}

// =========================================================
// IN-MEMORY STR PATH
// When dataset fits in RAM: 1 read, sort in-place, 1 write.
// Eliminates all intermediate disk I/O.
// =========================================================

void external_str_inmemory(const std::string& input,
                           const std::string& output,
                           size_t total_points) {

    size_t total_bytes = total_points * sizeof(Point);
    size_t chunk = SORT_CHUNK_POINTS;
    size_t slice = compute_str_slice_size(total_points);

    std::cout << "\n*** IN-MEMORY PATH (dataset fits in RAM) ***\n";
    std::cout << "STR slice size:     " << slice << " points ("
              << slice * sizeof(Point) / (1024.0*1024.0) << " MB)\n";

    auto wall_start = Clock::now();

    // ==========================================================
    // Step 1: Read entire input file into memory
    // ==========================================================
    std::vector<Point> data(total_points);

    auto t_read0 = Clock::now();
    {
        std::ifstream in(input, std::ios::binary);
        in.read(reinterpret_cast<char*>(data.data()), total_bytes);
        in.close();
    }
    auto t_read1 = Clock::now();
    double read_ms = Ms(t_read1 - t_read0).count();

    // ==========================================================
    // Step 2: GPU sort each chunk by X
    // ==========================================================
    g_stats = {};

    // Pre-allocate device buffer once (reused for every chunk)
    Point* d_buf = nullptr;
    auto ta0 = Clock::now();
    cudaMalloc(&d_buf, chunk * sizeof(Point));
    cudaDeviceSynchronize();
    auto ta1 = Clock::now();
    g_stats.gpu_alloc_ms += Ms(ta1 - ta0).count();

    for (size_t i = 0; i < total_points; i += chunk) {
        size_t n = std::min(chunk, total_points - i);
        gpu_sort_reuse(data.data() + i, n, true, d_buf);
    }
    TimingStats xsort_stats = g_stats;
    xsort_stats.print("Step 2 \u2014 GPU sort chunks by X");

    // ==========================================================
    // Step 3: In-memory merge of X-sorted runs
    //         Iterative pairwise std::inplace_merge (no extra I/O)
    // ==========================================================
    auto t_merge0 = Clock::now();

    auto cmpX = [](const Point& a, const Point& b) { return a.x < b.x; };
    for (size_t width = chunk; width < total_points; width *= 2) {
        size_t pairs = 0;
        for (size_t i = 0; i + width < total_points; i += 2 * width) {
            size_t mid = i + width;
            size_t end = std::min(i + 2 * width, total_points);
            std::inplace_merge(data.data() + i, data.data() + mid,
                               data.data() + end, cmpX);
            pairs++;
        }
        std::cout << "  Merge pass width=" << width
                  << "  pairs=" << pairs << std::endl;
    }

    auto t_merge1 = Clock::now();
    double merge_ms = Ms(t_merge1 - t_merge0).count();
    std::cout << "\n===== Timing: Step 3 \u2014 In-memory merge =====\n"
              << std::fixed << std::setprecision(2)
              << "  CPU merge:   " << merge_ms << " ms\n";

    // ==========================================================
    // Step 4: GPU sort each STR slice by Y
    // ==========================================================
    g_stats = {};

    for (size_t i = 0; i < total_points; i += slice) {
        size_t n = std::min(slice, total_points - i);
        gpu_sort_reuse(data.data() + i, n, false, d_buf);
    }
    TimingStats ysort_stats = g_stats;
    ysort_stats.print("Step 4 \u2014 GPU sort slices by Y");

    cudaFree(d_buf);

    // ==========================================================
    // Step 5: Build R-Tree from STR-sorted points
    // ==========================================================
    auto t_tree0 = Clock::now();
    uint32_t height;
    std::vector<RTreeNode> nodes = build_rtree(data.data(), total_points,
                                               RTREE_LEAF_CAPACITY,
                                               RTREE_INTERNAL_CAPACITY,
                                               height);
    auto t_tree1 = Clock::now();
    double tree_ms = Ms(t_tree1 - t_tree0).count();

    std::cout << "\n===== Timing: Step 5 \u2014 Build R-Tree =====\n"
              << std::fixed << std::setprecision(2)
              << "  Tree build:  " << tree_ms << " ms\n"
              << "  Nodes:       " << nodes.size() << "\n"
              << "  Height:      " << height << "\n"
              << "  Leaf nodes:  "
              << ((total_points + RTREE_LEAF_CAPACITY - 1) / RTREE_LEAF_CAPACITY)
              << "\n";

    // ==========================================================
    // Step 6: Write R-Tree to output file
    // ==========================================================
    auto t_write0 = Clock::now();
    write_rtree_file(output, nodes, height, data.data(), total_points);
    auto t_write1 = Clock::now();
    double write_ms = Ms(t_write1 - t_write0).count();

    // ==========================================================
    // Summary
    // ==========================================================
    auto wall_end = Clock::now();
    double wall_ms = Ms(wall_end - wall_start).count();

    size_t out_bytes = sizeof(RTreeHeader)
                     + nodes.size() * sizeof(RTreeNode)
                     + total_bytes;

    auto bw = [](size_t bytes, double ms) -> double {
        return (ms > 0) ? (bytes / (1024.0*1024.0)) / (ms / 1000.0) : 0;
    };

    std::cout << "\n===== Timing: GRAND TOTAL (in-memory) =====\n"
              << std::fixed << std::setprecision(2)
              << "  Disk read:      " << read_ms << " ms  ("
              << bw(total_bytes, read_ms) << " MB/s, "
              << total_bytes / (1024.0*1024.0) << " MB)\n"
              << "  GPU X-sort:     "
              << (xsort_stats.h2d_ms + xsort_stats.gpu_compute_ms
                  + xsort_stats.d2h_ms + xsort_stats.gpu_alloc_ms) << " ms\n"
              << "    H2D:          " << xsort_stats.h2d_ms << " ms\n"
              << "    Compute:      " << xsort_stats.gpu_compute_ms << " ms\n"
              << "    D2H:          " << xsort_stats.d2h_ms << " ms\n"
              << "  CPU merge:      " << merge_ms << " ms\n"
              << "  GPU Y-sort:     "
              << (ysort_stats.h2d_ms + ysort_stats.gpu_compute_ms
                  + ysort_stats.d2h_ms) << " ms\n"
              << "    H2D:          " << ysort_stats.h2d_ms << " ms\n"
              << "    Compute:      " << ysort_stats.gpu_compute_ms << " ms\n"
              << "    D2H:          " << ysort_stats.d2h_ms << " ms\n"
              << "  R-Tree build:   " << tree_ms << " ms\n"
              << "  Disk write:     " << write_ms << " ms  ("
              << bw(out_bytes, write_ms) << " MB/s, "
              << out_bytes / (1024.0*1024.0) << " MB)\n"
              << "  TOTAL:          " << wall_ms << " ms\n"
              << "\nTotal wall-clock time: " << wall_ms / 1000.0 << " s\n"
              << "STR R-Tree (in-memory) complete." << std::endl;
}

// =========================================================
// DISK-BASED STR PATH (triple-buffered, for large datasets)
//
// Optimizations over naive approach:
//   - K-way merge in Phase 1b (single pass instead of log2(k))
//   - Leaf MBR computation fused into Phase 2 Y-sort pipeline
//   - Points written directly to output file (no intermediate temp)
//   - Pre-computed R-tree layout allows seeking in output
//   - Run files deleted as consumed during merge
// =========================================================

void external_str_disk(const std::string& input,
                       const std::string& output,
                       size_t total_points) {

    std::cout << "\n*** DISK PATH (dataset exceeds RAM budget) ***\n";

    size_t slice = compute_str_slice_size(total_points);

    std::cout << "STR slice size:     " << slice << " points ("
              << slice * sizeof(Point) / (1024.0*1024.0) << " MB)\n";

    const std::string sorted_x = "tmp/sorted_x.bin";

    // Ensure tmp/ exists
    std::filesystem::create_directories("tmp");

    auto wall_start = Clock::now();

    // =====================================================
    // 1) GLOBAL SORT BY X (generate runs + k-way merge)
    // =====================================================

    g_stats = {};  // reset for phase 1
    auto phase1_start = Clock::now();

    size_t runs = generate_runs(input, SORT_CHUNK_POINTS, true, "tmp/x_run_");
    TimingStats gen_stats = g_stats;
    gen_stats.print("Phase 1a \u2014 Generate X-sorted runs");

    g_stats = {};
    kway_merge_files(runs, "tmp/x_run_", sorted_x, true);
    TimingStats merge_stats = g_stats;
    merge_stats.print("Phase 1b \u2014 K-way merge X-sorted runs");

    auto phase1_end = Clock::now();
    std::cout << "Phase 1 wall time: "
              << Ms(phase1_end - phase1_start).count() << " ms\n";

    // =====================================================
    // 2) STR TILING + R-TREE BUILD + WRITE OUTPUT
    //
    // Fused pipeline: reads X-sorted data from sorted_x,
    // Y-sorts each slice on GPU, computes leaf MBRs inline,
    // and writes sorted points directly to the output file.
    // Eliminates the intermediate sorted_str.tmp entirely.
    // =====================================================

    g_stats = {};  // reset for phase 2
    auto phase2_start = Clock::now();

    // Pre-compute R-tree layout (purely structural, no data needed)
    size_t num_nodes   = precompute_num_nodes(total_points,
                                               RTREE_LEAF_CAPACITY,
                                               RTREE_INTERNAL_CAPACITY);
    size_t num_leaves  = (total_points + RTREE_LEAF_CAPACITY - 1) / RTREE_LEAF_CAPACITY;
    size_t points_offset = sizeof(RTreeHeader) + num_nodes * sizeof(RTreeNode);

    std::cout << "  Pre-computed: " << num_nodes << " total nodes, "
              << num_leaves << " leaves, points offset = "
              << points_offset / (1024.0*1024.0) << " MB\n";

    std::vector<RTreeNode> nodes;
    nodes.reserve(num_nodes);

    size_t slice_bytes = slice * sizeof(Point);

    // 3 pinned host buffers for triple-buffered Y-sort pipeline
    Point* sbufs[3];
    for (int i = 0; i < 3; i++)
        cudaMallocHost(&sbufs[i], slice_bytes);

    size_t points_written = 0;  // tracks global point offset for leaf MBRs

    {
        std::ifstream in(sorted_x, std::ios::binary);

        // Open output file — reserve space for header+nodes at the front.
        // We'll seek back and write them after all points are in place.
        std::ofstream out(output, std::ios::binary);
        {
            constexpr size_t ZBUF_SZ = 1024 * 1024;  // 1 MB chunks
            std::vector<char> zbuf(std::min(points_offset, ZBUF_SZ), 0);
            size_t remaining = points_offset;
            while (remaining > 0) {
                size_t w = std::min(remaining, ZBUF_SZ);
                out.write(zbuf.data(), w);
                remaining -= w;
            }
        }
        // File cursor now at points_offset — ready for point data

        // Pre-read first slice
        auto tr0 = Clock::now();
        in.read(reinterpret_cast<char*>(sbufs[0]), slice_bytes);
        size_t n0 = in.gcount() / sizeof(Point);
        auto tr1 = Clock::now();
        g_stats.disk_read_ms += Ms(tr1 - tr0).count();
        g_stats.bytes_read   += n0 * sizeof(Point);

        int    sort_idx    = 0;
        size_t sort_n      = n0;
        int    read_idx    = 1;
        int    write_idx   = -1;
        size_t write_n     = 0;
        bool   input_done  = (n0 == 0);

        while (sort_n > 0) {
            // 1) Async read next slice from sorted_x
            std::future<std::pair<size_t, double>> read_fut;
            bool do_read = !input_done;
            if (do_read) {
                Point* rp = sbufs[read_idx];
                read_fut = std::async(std::launch::async,
                    [&in, rp, slice_bytes]() -> std::pair<size_t, double> {
                        auto t0 = Clock::now();
                        in.read(reinterpret_cast<char*>(rp), slice_bytes);
                        size_t n = in.gcount() / sizeof(Point);
                        auto t1 = Clock::now();
                        return {n, Ms(t1 - t0).count()};
                    });
            }

            // 2) Async write previous Y-sorted slice directly to output
            std::future<double> write_fut;
            bool do_write = (write_idx >= 0);
            if (do_write) {
                Point* wp = sbufs[write_idx];
                size_t wn = write_n;
                write_fut = std::async(std::launch::async,
                    [&out, wp, wn]() -> double {
                        auto t0 = Clock::now();
                        out.write(reinterpret_cast<char*>(wp),
                                  wn * sizeof(Point));
                        auto t1 = Clock::now();
                        return Ms(t1 - t0).count();
                    });
            }

            // 3) GPU sort current slice by Y (overlaps with I/O)
            gpu_sort(sbufs[sort_idx], sort_n, false);

            // 4) Compute leaf MBRs for the just-sorted slice (data is cache-hot)
            {
                size_t cap = RTREE_LEAF_CAPACITY;
                size_t nleaves = (sort_n + cap - 1) / cap;
                for (size_t li = 0; li < nleaves; li++) {
                    size_t start = li * cap;
                    size_t cnt = std::min(cap, sort_n - start);

                    MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
                    for (size_t j = 0; j < cnt; j++) {
                        const Point& p = sbufs[sort_idx][start + j];
                        m.min_x = std::min(m.min_x, p.x);
                        m.min_y = std::min(m.min_y, p.y);
                        m.max_x = std::max(m.max_x, p.x);
                        m.max_y = std::max(m.max_y, p.y);
                    }

                    RTreeNode nd{};
                    nd.mbr          = m;
                    nd.first_child  = points_written + start;
                    nd.num_children = (uint32_t)cnt;
                    nd.is_leaf      = 1;
                    nodes.push_back(nd);
                }
                points_written += sort_n;
            }

            // 5) Collect async results
            size_t next_n = 0;
            if (do_read) {
                std::pair<size_t, double> res = read_fut.get();
                g_stats.disk_read_ms += res.second;
                g_stats.bytes_read   += res.first * sizeof(Point);
                next_n = res.first;
                if (res.first == 0) input_done = true;
            }
            if (do_write) {
                double ms = write_fut.get();
                g_stats.disk_write_ms += ms;
                g_stats.bytes_written += write_n * sizeof(Point);
            }

            // 6) Rotate buffers
            write_idx = sort_idx;
            write_n   = sort_n;
            sort_n    = next_n;
            if (sort_n > 0) {
                sort_idx = read_idx;
                for (int i = 0; i < 3; i++) {
                    if (i != sort_idx && i != write_idx) {
                        read_idx = i;
                        break;
                    }
                }
            }
        }

        // Flush last sorted slice to output
        if (write_idx >= 0) {
            auto tw0 = Clock::now();
            out.write(reinterpret_cast<char*>(sbufs[write_idx]),
                      write_n * sizeof(Point));
            auto tw1 = Clock::now();
            g_stats.disk_write_ms += Ms(tw1 - tw0).count();
            g_stats.bytes_written += write_n * sizeof(Point);
        }

        in.close();
        out.close();
    }

    for (int i = 0; i < 3; i++) cudaFreeHost(sbufs[i]);

    // Build internal R-tree levels from the leaf nodes computed inline
    auto t_tree0 = Clock::now();
    uint32_t height;
    assert(nodes.size() == num_leaves);
    build_internal_levels(nodes, num_leaves, RTREE_INTERNAL_CAPACITY, height);
    assert(nodes.size() == num_nodes);
    auto t_tree1 = Clock::now();
    double tree_ms = Ms(t_tree1 - t_tree0).count();

    // Write header + nodes at the beginning of the output file
    // (points are already in place after the reserved space)
    auto t_hdr0 = Clock::now();
    {
        RTreeHeader hdr{};
        hdr.magic             = 0x52545245;  // "RTRE"
        hdr.leaf_capacity     = (uint32_t)RTREE_LEAF_CAPACITY;
        hdr.height            = height;
        hdr.internal_capacity = (uint32_t)RTREE_INTERNAL_CAPACITY;
        hdr.num_points        = total_points;
        hdr.num_nodes         = nodes.size();

        std::fstream fout(output, std::ios::binary | std::ios::in | std::ios::out);
        fout.seekp(0);
        fout.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        fout.write(reinterpret_cast<const char*>(nodes.data()),
                   nodes.size() * sizeof(RTreeNode));
        fout.close();
    }
    auto t_hdr1 = Clock::now();
    double hdr_ms = Ms(t_hdr1 - t_hdr0).count();

    std::cout << "  R-Tree written: " << nodes.size() << " nodes, "
              << height << " levels, root MBR = ["
              << nodes.back().mbr.min_x << ", "
              << nodes.back().mbr.min_y << "] x ["
              << nodes.back().mbr.max_x << ", "
              << nodes.back().mbr.max_y << "]\n";

    // Clean up temp files
    std::filesystem::remove(sorted_x);

    auto phase2_end = Clock::now();
    TimingStats tile_stats = g_stats;
    tile_stats.print("Phase 2 \u2014 STR Y-sort + R-Tree build + direct write");
    std::cout << "Phase 2 wall time: "
              << Ms(phase2_end - phase2_start).count() << " ms\n";

    // =====================================================
    // Grand total
    // =====================================================
    TimingStats total;
    total += gen_stats;
    total += merge_stats;
    total += tile_stats;
    total.print("GRAND TOTAL (I/O + GPU)");

    size_t out_bytes = sizeof(RTreeHeader)
                     + nodes.size() * sizeof(RTreeNode)
                     + total_points * sizeof(Point);

    auto wall_end = Clock::now();
    std::cout << std::fixed << std::setprecision(2)
              << "  R-Tree build:      " << tree_ms << " ms  ("
              << num_nodes << " nodes)\n"
              << "  Header+nodes write:" << hdr_ms << " ms  ("
              << points_offset / (1024.0*1024.0) << " MB)\n"
              << "\nTotal wall-clock time: "
              << Ms(wall_end - wall_start).count() / 1000.0
              << " s\n";
    std::cout << "STR R-Tree (external) complete." << std::endl;
}

// =========================================================
// DISPATCHER: chooses in-memory or disk path
// =========================================================

void external_str(const std::string& input,
                  const std::string& output) {

    size_t file_size = std::filesystem::file_size(input);
    size_t total_points = file_size / sizeof(Point);

    std::cout << "Total points:       " << total_points << std::endl;
    std::cout << "USABLE_GPU_BYTES:   " << USABLE_GPU_BYTES / (1024*1024)
              << " MB" << std::endl;
    std::cout << "USABLE_RAM_BYTES:   " << USABLE_RAM_BYTES / (1024*1024)
              << " MB" << std::endl;
    std::cout << "SORT_CHUNK_POINTS:  " << SORT_CHUNK_POINTS << std::endl;
    std::cout << "RTREE_NODE_BYTES:   " << RTREE_NODE_BYTES << std::endl;
    std::cout << "LEAF_CAPACITY:      " << RTREE_LEAF_CAPACITY << std::endl;
    std::cout << "INTERNAL_CAPACITY:  " << RTREE_INTERNAL_CAPACITY << std::endl;
    std::cout << "STR slice size:     " << compute_str_slice_size(total_points)
              << std::endl;
    std::cout << "Dataset size:       " << file_size / (1024*1024)
              << " MB" << std::endl;

    if (file_size <= USABLE_RAM_BYTES) {
        external_str_inmemory(input, output, total_points);
    } else {
        external_str_disk(input, output, total_points);
    }
}

// =========================================================
// MAIN
// =========================================================

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Usage: ./external_str input.bin output.bin" << std::endl;
        return 0;
    }

    external_str(argv[1], argv[2]);

    return 0;
}
