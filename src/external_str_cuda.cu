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

struct RTreeHeader {
    uint32_t magic;          // 0x52545245  "RTRE"
    uint32_t node_capacity;
    uint32_t height;
    uint32_t _pad;
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
// CPU MERGE (two sorted files) — streaming, with timing
// =========================================================

void merge_files(const std::string& fileA,
                 const std::string& fileB,
                 const std::string& output,
                 bool by_x) {

    // I/O buffer size (per stream). Merge is O(n) and I/O-bound,
    // so CPU is fine here — the GPU pays off in sort (O(n log n)).
    constexpr size_t IO_BUF = 16 * 1024 * 1024;  // 16M points = 128 MB

    std::ifstream fA(fileA, std::ios::binary);
    std::ifstream fB(fileB, std::ios::binary);
    std::ofstream fOut(output, std::ios::binary);

    std::vector<Point> bufA(IO_BUF), bufB(IO_BUF), bufOut(IO_BUF);

    size_t posA = 0, endA = 0;
    size_t posB = 0, endB = 0;
    size_t posOut = 0;
    bool eofA = false, eofB = false;

    auto refill = [&](std::ifstream& f, std::vector<Point>& buf,
                      size_t& pos, size_t& end, bool& eof) {
        auto t0 = Clock::now();
        f.read(reinterpret_cast<char*>(buf.data()), IO_BUF * sizeof(Point));
        end = f.gcount() / sizeof(Point);
        pos = 0;
        auto t1 = Clock::now();
        g_stats.disk_read_ms += Ms(t1 - t0).count();
        g_stats.bytes_read   += end * sizeof(Point);
        if (end == 0) eof = true;
    };

    auto flush_out = [&]() {
        if (posOut == 0) return;
        auto t0 = Clock::now();
        fOut.write(reinterpret_cast<char*>(bufOut.data()),
                   posOut * sizeof(Point));
        auto t1 = Clock::now();
        g_stats.disk_write_ms += Ms(t1 - t0).count();
        g_stats.bytes_written += posOut * sizeof(Point);
        posOut = 0;
    };

    auto cmp = [by_x](const Point& a, const Point& b) {
        return by_x ? (a.x < b.x) : (a.y < b.y);
    };

    // Initial fill
    refill(fA, bufA, posA, endA, eofA);
    refill(fB, bufB, posB, endB, eofB);

    // Two-way merge
    while (!eofA && !eofB) {
        if (cmp(bufA[posA], bufB[posB])) {
            bufOut[posOut++] = bufA[posA++];
            if (posA >= endA) refill(fA, bufA, posA, endA, eofA);
        } else {
            bufOut[posOut++] = bufB[posB++];
            if (posB >= endB) refill(fB, bufB, posB, endB, eofB);
        }
        if (posOut >= IO_BUF) flush_out();
    }

    // Drain remainder of A
    while (!eofA) {
        bufOut[posOut++] = bufA[posA++];
        if (posA >= endA) refill(fA, bufA, posA, endA, eofA);
        if (posOut >= IO_BUF) flush_out();
    }

    // Drain remainder of B
    while (!eofB) {
        bufOut[posOut++] = bufB[posB++];
        if (posB >= endB) refill(fB, bufB, posB, endB, eofB);
        if (posOut >= IO_BUF) flush_out();
    }

    flush_out();

    fA.close();
    fB.close();
    fOut.close();
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
// Pairwise GPU merge pass
// =========================================================

size_t merge_pass(size_t run_count,
                  const std::string& prefix_in,
                  const std::string& prefix_out,
                  bool by_x) {

    size_t new_runs = 0;

    for (size_t i = 0; i < run_count; i += 2) {
        std::string A = prefix_in + std::to_string(i) + ".bin";

        if (i + 1 >= run_count) {
            std::filesystem::rename(A,
                prefix_out + std::to_string(new_runs++) + ".bin");
            break;
        }

        std::string B = prefix_in + std::to_string(i + 1) + ".bin";
        std::string OUT = prefix_out + std::to_string(new_runs++) + ".bin";

        merge_files(A, B, OUT, by_x);
    }

    return new_runs;
}

// =========================================================
// STR slice size computation
// =========================================================

size_t compute_str_slice_size(size_t total_points) {
    size_t num_leaves = (total_points + RTREE_NODE_CAPACITY - 1) / RTREE_NODE_CAPACITY;
    size_t num_slices = (size_t)std::ceil(std::sqrt((double)num_leaves));
    size_t ideal_slice = (total_points + num_slices - 1) / num_slices;
    // Cap at GPU sort capacity
    return std::min(ideal_slice, (size_t)STR_MAX_SLICE);
}

// =========================================================
// R-Tree construction (bottom-up from STR-sorted points)
//
// Points must already be in STR order (sorted by X globally,
// then by Y within each vertical slice).  Consecutive groups
// of RTREE_NODE_CAPACITY points form leaf nodes.  Internal
// levels are built by grouping child MBRs.
// =========================================================

// --- Build from in-memory point array ---
std::vector<RTreeNode> build_rtree(const Point* points,
                                   size_t num_points,
                                   size_t capacity,
                                   uint32_t& height_out) {
    std::vector<RTreeNode> nodes;
    size_t num_leaves = (num_points + capacity - 1) / capacity;
    nodes.reserve(num_leaves * 2);  // rough upper bound

    // Leaf level
    for (size_t i = 0; i < num_leaves; i++) {
        size_t start = i * capacity;
        size_t cnt   = std::min(capacity, num_points - start);

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
        size_t next_count = (level_count + capacity - 1) / capacity;

        for (size_t i = 0; i < next_count; i++) {
            size_t s   = level_start + i * capacity;
            size_t cnt = std::min(capacity, level_count - i * capacity);

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

// --- Build from sorted file on disk (streams points for leaf MBRs) ---
std::vector<RTreeNode> build_rtree_from_file(const std::string& sorted_file,
                                             size_t num_points,
                                             size_t capacity,
                                             uint32_t& height_out) {
    std::vector<RTreeNode> nodes;
    size_t num_leaves = (num_points + capacity - 1) / capacity;
    nodes.reserve(num_leaves * 2);

    // Stream through sorted file to compute leaf MBRs
    std::ifstream in(sorted_file, std::ios::binary);
    std::vector<Point> buf(capacity);

    for (size_t i = 0; i < num_leaves; i++) {
        size_t cnt = std::min(capacity, num_points - i * capacity);
        in.read(reinterpret_cast<char*>(buf.data()), cnt * sizeof(Point));

        MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
        for (size_t j = 0; j < cnt; j++) {
            m.min_x = std::min(m.min_x, buf[j].x);
            m.min_y = std::min(m.min_y, buf[j].y);
            m.max_x = std::max(m.max_x, buf[j].x);
            m.max_y = std::max(m.max_y, buf[j].y);
        }

        RTreeNode nd{};
        nd.mbr          = m;
        nd.first_child  = i * capacity;
        nd.num_children = (uint32_t)cnt;
        nd.is_leaf      = 1;
        nodes.push_back(nd);
    }
    in.close();

    // Internal levels (identical to in-memory path)
    height_out = 1;
    size_t level_start = 0;
    size_t level_count = num_leaves;

    while (level_count > 1) {
        size_t next_count = (level_count + capacity - 1) / capacity;

        for (size_t i = 0; i < next_count; i++) {
            size_t s   = level_start + i * capacity;
            size_t cnt = std::min(capacity, level_count - i * capacity);

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
    hdr.magic         = 0x52545245;  // "RTRE"
    hdr.node_capacity = (uint32_t)RTREE_NODE_CAPACITY;
    hdr.height        = height;
    hdr.num_points    = num_points;
    hdr.num_nodes     = nodes.size();

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

// Variant: write header+nodes, then copy points from a file
void write_rtree_file_from_sorted(const std::string& path,
                                  const std::vector<RTreeNode>& nodes,
                                  uint32_t height,
                                  const std::string& sorted_points_file,
                                  size_t num_points) {
    RTreeHeader hdr{};
    hdr.magic         = 0x52545245;
    hdr.node_capacity = (uint32_t)RTREE_NODE_CAPACITY;
    hdr.height        = height;
    hdr.num_points    = num_points;
    hdr.num_nodes     = nodes.size();

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char*>(nodes.data()),
              nodes.size() * sizeof(RTreeNode));

    // Stream-copy sorted points
    constexpr size_t COPY_BUF = 16 * 1024 * 1024;  // 128 MB
    std::vector<Point> buf(COPY_BUF);
    std::ifstream in(sorted_points_file, std::ios::binary);
    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()), COPY_BUF * sizeof(Point));
        size_t n = in.gcount();
        if (n > 0) out.write(reinterpret_cast<char*>(buf.data()), n);
    }
    in.close();
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
                                               RTREE_NODE_CAPACITY, height);
    auto t_tree1 = Clock::now();
    double tree_ms = Ms(t_tree1 - t_tree0).count();

    std::cout << "\n===== Timing: Step 5 \u2014 Build R-Tree =====\n"
              << std::fixed << std::setprecision(2)
              << "  Tree build:  " << tree_ms << " ms\n"
              << "  Nodes:       " << nodes.size() << "\n"
              << "  Height:      " << height << "\n"
              << "  Leaf nodes:  "
              << ((total_points + RTREE_NODE_CAPACITY - 1) / RTREE_NODE_CAPACITY)
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
// =========================================================

void external_str_disk(const std::string& input,
                       const std::string& output,
                       size_t total_points) {

    std::cout << "\n*** DISK PATH (dataset exceeds RAM budget) ***\n";

    size_t slice = compute_str_slice_size(total_points);
    std::cout << "STR slice size:     " << slice << " points ("
              << slice * sizeof(Point) / (1024.0*1024.0) << " MB)\n";

    // Temp file for sorted points (tree goes into final output)
    const std::string sorted_tmp  = "tmp/sorted_str.tmp";
    const std::string sorted_x    = "tmp/sorted_x.bin";

    // Ensure tmp/ exists
    std::filesystem::create_directories("tmp");

    auto wall_start = Clock::now();

    // =====================================================
    // 1) GLOBAL SORT BY X (external, GPU merge)
    // =====================================================

    g_stats = {};  // reset for phase 1
    auto phase1_start = Clock::now();

    size_t runs = generate_runs(input, SORT_CHUNK_POINTS, true, "tmp/x_run_");
    TimingStats gen_stats = g_stats;
    gen_stats.print("Phase 1a \u2014 Generate X-sorted runs");

    g_stats = {};
    size_t merge_round = 0;
    while (runs > 1) {
        std::cout << "  Merge round " << merge_round++ << " : "
                  << runs << " runs" << std::endl;
        runs = merge_pass(runs, "tmp/x_run_", "tmp/x_tmp_", true);
    }
    TimingStats merge_stats = g_stats;
    merge_stats.print("Phase 1b \u2014 Merge X-sorted runs");

    auto phase1_end = Clock::now();
    std::cout << "Phase 1 wall time: "
              << Ms(phase1_end - phase1_start).count() << " ms\n";

    std::filesystem::rename("tmp/x_tmp_0.bin", sorted_x);

    // =====================================================
    // 2) STR TILING
    // Divide sorted_x into vertical slices
    // Each slice is sorted by Y independently
    // Write sorted points to temp file
    // =====================================================

    g_stats = {};  // reset for phase 2
    auto phase2_start = Clock::now();

    size_t slice_bytes = slice * sizeof(Point);

    // 3 pinned host buffers for triple-buffered Y-sort pipeline
    Point* sbufs[3];
    for (int i = 0; i < 3; i++)
        cudaMallocHost(&sbufs[i], slice_bytes);

    {
        std::ifstream in(sorted_x, std::ios::binary);
        std::ofstream out(sorted_tmp, std::ios::binary);

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
            // 1) Async read next slice
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

            // 2) Async write previous sorted slice
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

            // 5) Rotate buffers
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

        // Flush last sorted slice
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

    auto phase2_end = Clock::now();
    TimingStats tile_stats = g_stats;
    tile_stats.print("Phase 2 \u2014 STR Y-sort tiling");
    std::cout << "Phase 2 wall time: "
              << Ms(phase2_end - phase2_start).count() << " ms\n";

    // =====================================================
    // 3) BUILD R-TREE
    // Stream the sorted points file to compute leaf MBRs,
    // then build internal levels bottom-up in memory.
    // =====================================================

    auto phase3_start = Clock::now();
    uint32_t height;
    std::vector<RTreeNode> nodes = build_rtree_from_file(
        sorted_tmp, total_points, RTREE_NODE_CAPACITY, height);
    auto phase3_end = Clock::now();
    double tree_ms = Ms(phase3_end - phase3_start).count();

    std::cout << "\n===== Timing: Phase 3 \u2014 Build R-Tree =====\n"
              << std::fixed << std::setprecision(2)
              << "  Tree build:  " << tree_ms << " ms\n"
              << "  Nodes:       " << nodes.size() << "\n"
              << "  Height:      " << height << "\n"
              << "  Leaf nodes:  "
              << ((total_points + RTREE_NODE_CAPACITY - 1) / RTREE_NODE_CAPACITY)
              << "\n";

    // =====================================================
    // 4) WRITE FINAL R-TREE FILE
    // Header + Nodes + Points (copied from sorted temp)
    // =====================================================

    auto phase4_start = Clock::now();
    write_rtree_file_from_sorted(output, nodes, height, sorted_tmp, total_points);
    auto phase4_end = Clock::now();
    double write_ms = Ms(phase4_end - phase4_start).count();

    // Clean up temp file
    std::filesystem::remove(sorted_tmp);
    std::filesystem::remove(sorted_x);

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
              << "  R-Tree build:      " << tree_ms << " ms\n"
              << "  R-Tree file write: " << write_ms << " ms  ("
              << out_bytes / (1024.0*1024.0) << " MB)\n"
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
    std::cout << "RTREE_NODE_CAPACITY:" << RTREE_NODE_CAPACITY << std::endl;
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
