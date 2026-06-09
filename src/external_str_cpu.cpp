// external_str_cpu.cpp
// STR (Sort-Tile-Recursive) R-Tree bulk-loading — CPU-only version
// Drop-in replacement for external_str_cuda.cu for benchmarking
// without GPU — all sorting done with std::sort on the CPU.
//
// Output file format is identical:
//   [RTreeHeader  – 32 bytes]
//   [RTreeNode[]  – num_nodes × 32 bytes]
//   [Point[]      – num_points × 8 bytes]
//
// Root node is always nodes[num_nodes - 1].
// Leaf nodes reference contiguous ranges of points.
// Internal nodes reference contiguous ranges of child nodes.
//
// Compile: g++ -O3 -std=c++17 -pthread external_str_cpu.cpp -o external_str_cpu

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <cassert>
#include <chrono>
#include <future>
#include <algorithm>
#include <tuple>

#include "constants.h"

// =========================================================
// Timing infrastructure
// =========================================================

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// Threading contract for the global `g_stats` accumulator:
//   - The async I/O worker NEVER writes g_stats directly. It returns its
//     disk timings through a std::future; the MAIN thread folds them in.
//   - All CPU-sort fields are written only by the main thread.
//   Therefore no field is touched by two threads at once and no lock is
//   needed. Preserve this invariant if you add new timed work.
struct TimingStats {
    double disk_read_ms   = 0;
    double disk_write_ms  = 0;
    double cpu_sort_ms    = 0;

    // Phase 1b k-way merge — CPU heap-merge work.
    double cpu_merge_ms   = 0;

    // Phase 3 (single-threaded) work — folded into the grand total so it
    // reconciles with the total wall-clock time.
    double tree_ms        = 0;  // build internal R-tree levels
    double hdr_ms         = 0;  // header + nodes write

    // Wall-clock elapsed for the phase this struct describes. Reflects the
    // dual-thread overlap (CPU sort ‖ disk I/O); always <= COMPONENT SUM.
    double wall_ms        = 0;

    size_t bytes_read     = 0;
    size_t bytes_written  = 0;

    double component_sum_ms() const {
        return disk_read_ms + disk_write_ms + cpu_sort_ms
             + cpu_merge_ms + tree_ms + hdr_ms;
    }

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
                  << "  CPU sort:      " << cpu_sort_ms << " ms\n";
        if (cpu_merge_ms > 0)
            std::cout << "  CPU merge:     " << cpu_merge_ms << " ms\n";
        if (tree_ms > 0 || hdr_ms > 0)
            std::cout << "  Tree build:    " << tree_ms     << " ms\n"
                      << "  Header write:  " << hdr_ms      << " ms\n";

        double comp = component_sum_ms();
        std::cout << "  COMPONENT SUM: " << comp << " ms  (serialized cost)\n";
        if (wall_ms > 0) {
            std::cout << "  WALL:          " << wall_ms << " ms  (actual elapsed)\n"
                      << "  OVERLAP SAVED: "
                      << (comp > wall_ms ? comp - wall_ms : 0.0)
                      << " ms  (CPU ‖ I/O concurrency)\n";
        }
    }

    TimingStats& operator+=(const TimingStats& o) {
        disk_read_ms   += o.disk_read_ms;
        disk_write_ms  += o.disk_write_ms;
        cpu_sort_ms    += o.cpu_sort_ms;
        cpu_merge_ms   += o.cpu_merge_ms;
        tree_ms        += o.tree_ms;
        hdr_ms         += o.hdr_ms;
        bytes_read     += o.bytes_read;
        bytes_written  += o.bytes_written;
        return *this;
    }
};

TimingStats g_stats;  // global accumulator (see threading contract above)

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
    bool operator()(const Point& a, const Point& b) const {
        return a.x < b.x;
    }
};

struct CompareY {
    bool operator()(const Point& a, const Point& b) const {
        return a.y < b.y;
    }
};

// =========================================================
// CPU SORT (chunk local) — with detailed timing
// =========================================================

void cpu_sort(Point* data, size_t n, bool by_x) {
    auto t0 = Clock::now();
    if (by_x)
        std::sort(data, data + n, CompareX());
    else
        std::sort(data, data + n, CompareY());
    auto t1 = Clock::now();
    g_stats.cpu_sort_ms += Ms(t1 - t0).count();
}

// =========================================================
// Cached run: a sorted chunk either in RAM or on disk.
// =========================================================

struct CachedRun {
    std::vector<Point> data;  // non-empty → cached in RAM
    std::string path;         // non-empty → stored on disk
    size_t num_points = 0;
};

// =========================================================
// Generate X-sorted runs with dual-thread pipeline.
//
// Main thread:  CPU sort
// I/O thread:   serialized write-then-read (no concurrent disk ops)
//
// Runs that fit within ram_budget are cached in host RAM;
// the rest are written to disk as tmp/x_run_N.bin.
// =========================================================

void generate_runs_cached(const std::string& input,
                          size_t points_per_chunk,
                          std::vector<CachedRun>& runs,
                          size_t& ram_budget) {
    // A chunk never needs to be larger than the whole input. Cap the buffers
    // so small datasets don't allocate SORT_CHUNK_POINTS-sized host buffers.
    size_t file_points = std::filesystem::file_size(input) / sizeof(Point);
    size_t eff_chunk_points = std::min(points_per_chunk,
                                       std::max<size_t>(file_points, 1));
    size_t chunk_bytes = eff_chunk_points * sizeof(Point);

    // 2 host buffers for dual-thread pipeline
    std::vector<Point> buf0(eff_chunk_points);
    std::vector<Point> buf1(eff_chunk_points);
    Point* bufs[2] = { buf0.data(), buf1.data() };

    std::ifstream in(input, std::ios::binary);

    // Pre-read first chunk synchronously
    auto t0 = Clock::now();
    in.read(reinterpret_cast<char*>(bufs[0]), chunk_bytes);
    size_t n0 = in.gcount() / sizeof(Point);
    auto t1 = Clock::now();
    g_stats.disk_read_ms += Ms(t1 - t0).count();
    g_stats.bytes_read   += n0 * sizeof(Point);

    if (n0 == 0) {
        in.close();
        return;
    }

    int sort_buf = 0, io_buf = 1;
    size_t sort_n = n0;
    bool input_done = false;

    // Pending disk write from previous iteration
    Point*      pw_data = nullptr;
    size_t      pw_n    = 0;
    std::string pw_path;

    while (sort_n > 0) {
        // Snapshot pending-write state for the I/O thread
        Point*      wd       = pw_data;
        size_t      wn       = pw_n;
        std::string wp       = pw_path;
        bool        do_read  = !input_done;
        Point*      read_dst = bufs[io_buf];

        // --- I/O thread: serialize write (previous) then read (next) ---
        std::future<std::tuple<size_t, double, double>> io_fut;
        bool do_io = (wd != nullptr) || do_read;

        if (do_io) {
            io_fut = std::async(std::launch::async,
                [wd, wn, wp, read_dst, chunk_bytes, &in, do_read]()
                    -> std::tuple<size_t, double, double> {
                    double write_ms = 0, read_ms = 0;
                    size_t nr = 0;

                    if (wd) {
                        auto tw0 = Clock::now();
                        std::ofstream f(wp, std::ios::binary);
                        f.write(reinterpret_cast<const char*>(wd),
                                wn * sizeof(Point));
                        f.close();
                        write_ms = Ms(Clock::now() - tw0).count();
                    }

                    if (do_read) {
                        auto tr0 = Clock::now();
                        in.read(reinterpret_cast<char*>(read_dst),
                                chunk_bytes);
                        nr = in.gcount() / sizeof(Point);
                        read_ms = Ms(Clock::now() - tr0).count();
                    }

                    return {nr, write_ms, read_ms};
                });
        }

        // --- Main thread: CPU sort ---
        cpu_sort(bufs[sort_buf], sort_n, true);

        // Decide: cache in RAM or defer disk write
        size_t run_bytes = sort_n * sizeof(Point);
        CachedRun run;
        run.num_points = sort_n;

        if (ram_budget >= run_bytes) {
            run.data.assign(bufs[sort_buf], bufs[sort_buf] + sort_n);
            ram_budget -= run_bytes;
            pw_data = nullptr;
        } else {
            run.path = "tmp/x_run_" + std::to_string(runs.size()) + ".bin";
            pw_data = bufs[sort_buf];
            pw_n    = sort_n;
            pw_path = run.path;
        }
        runs.push_back(std::move(run));

        // Collect I/O results
        size_t next_n = 0;
        if (do_io) {
            auto [nr, wms, rms] = io_fut.get();
            g_stats.disk_write_ms += wms;
            if (wd) g_stats.bytes_written += wn * sizeof(Point);
            g_stats.disk_read_ms  += rms;
            g_stats.bytes_read    += nr * sizeof(Point);
            next_n = nr;
            if (nr == 0 && do_read) input_done = true;
        }

        // Swap buffers
        std::swap(sort_buf, io_buf);
        sort_n = next_n;
    }

    // Flush last pending disk write
    if (pw_data) {
        auto tw0 = Clock::now();
        std::ofstream f(pw_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(pw_data),
                pw_n * sizeof(Point));
        f.close();
        auto tw1 = Clock::now();
        g_stats.disk_write_ms += Ms(tw1 - tw0).count();
        g_stats.bytes_written += pw_n * sizeof(Point);
    }

    in.close();
}

// =========================================================
// Unified RunReader: reads points from a CachedRun
// (either memory-backed or file-backed).
// =========================================================

struct RunReader {
    // Memory source
    const Point* mem_ptr = nullptr;
    size_t mem_size = 0;
    size_t mem_pos  = 0;

    // File source
    std::ifstream file;
    std::vector<Point> buf;
    size_t buf_pos = 0, buf_end = 0;
    std::string file_path;

    bool is_mem   = false;
    bool eof_flag = false;

    void open(const CachedRun& run, size_t file_buf_sz) {
        if (!run.data.empty()) {
            is_mem   = true;
            mem_ptr  = run.data.data();
            mem_size = run.num_points;
            mem_pos  = 0;
            eof_flag = (mem_size == 0);
        } else {
            is_mem    = false;
            file_path = run.path;
            buf.resize(file_buf_sz);
            file.open(run.path, std::ios::binary);
            refill();
        }
    }

    void refill() {
        if (eof_flag || is_mem) return;
        auto t0 = Clock::now();
        file.read(reinterpret_cast<char*>(buf.data()),
                  buf.size() * sizeof(Point));
        buf_end = file.gcount() / sizeof(Point);
        buf_pos = 0;
        auto t1 = Clock::now();
        g_stats.disk_read_ms += Ms(t1 - t0).count();
        g_stats.bytes_read   += buf_end * sizeof(Point);
        if (buf_end == 0) eof_flag = true;
    }

    bool eof() const { return eof_flag; }

    const Point& current() const {
        if (is_mem) return mem_ptr[mem_pos];
        return buf[buf_pos];
    }

    bool advance() {
        if (is_mem) {
            mem_pos++;
            if (mem_pos >= mem_size) { eof_flag = true; return false; }
            return true;
        }
        buf_pos++;
        if (buf_pos >= buf_end) {
            refill();
            if (eof_flag) return false;
        }
        return true;
    }

    void close_and_delete() {
        if (!is_mem) {
            file.close();
            if (!file_path.empty())
                std::filesystem::remove(file_path);
        }
    }
};

// =========================================================
// K-way merge of cached runs.
//
// If all runs are in RAM, merges into mem_output (no disk I/O).
// Otherwise merges to disk_output file.
// Sets all_in_memory accordingly.
// =========================================================

void kway_merge(std::vector<CachedRun>& runs,
                const std::string& disk_output,
                std::vector<Point>& mem_output,
                bool& all_in_memory,
                size_t total_points,
                bool by_x) {
    size_t run_count = runs.size();
    if (run_count == 0) { all_in_memory = true; return; }

    bool all_cached = true;
    for (auto& r : runs)
        if (r.data.empty()) { all_cached = false; break; }
    all_in_memory = all_cached;

    // Buffer size per file-backed reader
    constexpr size_t MAX_MERGE_MEM  = 2ULL * 1024 * 1024 * 1024;
    constexpr size_t MIN_BUF_POINTS = 4096;
    constexpr size_t MAX_BUF_POINTS = 16 * 1024 * 1024;

    size_t buf_points = MAX_MERGE_MEM / ((run_count + 1) * sizeof(Point));
    buf_points = std::max(buf_points, MIN_BUF_POINTS);
    buf_points = std::min(buf_points, MAX_BUF_POINTS);

    std::cout << "  K-way merge: " << run_count << " runs"
              << (all_cached ? " (all in RAM)" : " (some on disk)")
              << ", " << buf_points << " pts/buf\n";

    // Open unified readers
    std::vector<RunReader> readers(run_count);
    for (size_t i = 0; i < run_count; i++)
        readers[i].open(runs[i], buf_points);

    // Min-heap
    struct HeapEntry {
        Point  pt;
        size_t run;
    };
    auto cmp = [by_x](const HeapEntry& a, const HeapEntry& b) {
        return by_x ? (a.pt.x > b.pt.x) : (a.pt.y > b.pt.y);
    };

    std::vector<HeapEntry> heap;
    heap.reserve(run_count);
    for (size_t i = 0; i < run_count; i++) {
        if (!readers[i].eof())
            heap.push_back({readers[i].current(), i});
    }
    std::make_heap(heap.begin(), heap.end(), cmp);

    if (all_in_memory) {
        // Merge to memory — no disk I/O, pure CPU heap merge.
        auto t0 = Clock::now();
        mem_output.resize(total_points);
        size_t pos = 0;

        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            HeapEntry top = heap.back();
            heap.pop_back();

            mem_output[pos++] = top.pt;

            RunReader& r = readers[top.run];
            if (r.advance()) {
                heap.push_back({r.current(), top.run});
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else {
                r.close_and_delete();
            }
        }
        g_stats.cpu_merge_ms += Ms(Clock::now() - t0).count();
    } else {
        // Merge to file. Disk writes are timed in flush(); the rest of the
        // loop is CPU heap-merge work, recovered as elapsed minus disk time.
        auto merge_t0 = Clock::now();
        double disk_before = g_stats.disk_write_ms;
        std::vector<Point> out_buf(buf_points);
        size_t out_pos = 0;
        std::ofstream fout(disk_output, std::ios::binary);

        auto flush = [&]() {
            if (out_pos == 0) return;
            auto t0 = Clock::now();
            fout.write(reinterpret_cast<char*>(out_buf.data()),
                       out_pos * sizeof(Point));
            auto t1 = Clock::now();
            g_stats.disk_write_ms += Ms(t1 - t0).count();
            g_stats.bytes_written += out_pos * sizeof(Point);
            out_pos = 0;
        };

        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            HeapEntry top = heap.back();
            heap.pop_back();

            out_buf[out_pos++] = top.pt;
            if (out_pos >= buf_points) flush();

            RunReader& r = readers[top.run];
            if (r.advance()) {
                heap.push_back({r.current(), top.run});
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else {
                r.close_and_delete();
            }
        }

        flush();
        fout.close();
        double elapsed = Ms(Clock::now() - merge_t0).count();
        g_stats.cpu_merge_ms += elapsed - (g_stats.disk_write_ms - disk_before);
    }

    // Free cached run data
    for (auto& r : runs) {
        r.data.clear();
        r.data.shrink_to_fit();
    }
}

// =========================================================
// STR slice size computation
// =========================================================

size_t compute_str_slice_size(size_t total_points) {
    size_t leaf_cap = RTREE_LEAF_CAPACITY;
    size_t num_leaves = (total_points + leaf_cap - 1) / leaf_cap;
    size_t num_slices = (size_t)std::ceil(std::sqrt((double)num_leaves));

    size_t leaves_per_slice = (num_leaves + num_slices - 1) / num_slices;
    size_t slice_points = leaves_per_slice * leaf_cap;

    // Cap at SORT_CHUNK_POINTS (same budget, but now CPU RAM)
    return std::min(slice_points, STR_MAX_SLICE);
}

// =========================================================
// Pre-compute total number of R-tree nodes.
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
// STR sort a contiguous range of RTreeNode entries.
// =========================================================

void sort_nodes_str(std::vector<RTreeNode>& nodes,
                    size_t level_start,
                    size_t level_count,
                    size_t group_cap) {
    if (level_count <= group_cap) return;

    auto begin = nodes.begin() + level_start;

    size_t num_groups = (level_count + group_cap - 1) / group_cap;
    size_t num_slices = (size_t)std::ceil(std::sqrt((double)num_groups));

    size_t nodes_per_slice = (level_count + num_slices - 1) / num_slices;
    nodes_per_slice = ((nodes_per_slice + group_cap - 1) / group_cap) * group_cap;

    // 1) Sort entire level by MBR center X
    std::sort(begin, begin + level_count,
              [](const RTreeNode& a, const RTreeNode& b) {
                  return (a.mbr.min_x + a.mbr.max_x) <
                         (b.mbr.min_x + b.mbr.max_x);
              });

    // 2) Within each vertical slice, sort by MBR center Y
    for (size_t off = 0; off < level_count; off += nodes_per_slice) {
        size_t cnt = std::min(nodes_per_slice, level_count - off);
        std::sort(begin + off, begin + off + cnt,
                  [](const RTreeNode& a, const RTreeNode& b) {
                      return (a.mbr.min_y + a.mbr.max_y) <
                             (b.mbr.min_y + b.mbr.max_y);
                  });
    }
}

// =========================================================
// Build internal R-tree levels from a vector of leaf nodes.
// =========================================================

void build_internal_levels(std::vector<RTreeNode>& nodes,
                           size_t num_leaves,
                           size_t internal_cap,
                           uint32_t& height_out) {
    height_out = 1;
    size_t level_start = 0;
    size_t level_count = num_leaves;

    while (level_count > 1) {
        sort_nodes_str(nodes, level_start, level_count, internal_cap);

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
// Unified STR R-Tree builder — CPU-only version.
//
// Phase 1a: CPU X-sort chunks (dual-thread: CPU sort ‖ serial I/O)
// Phase 1b: K-way merge (in-memory if all cached, disk otherwise)
// Phase 2:  CPU Y-sort STR slices + leaf MBR computation
//           (dual-thread: CPU sort ‖ serial I/O)
// Phase 3:  Build internal R-tree levels + write header
// =========================================================

void external_str_build(const std::string& input,
                        const std::string& output,
                        size_t total_points) {

    size_t slice = compute_str_slice_size(total_points);

    std::filesystem::create_directories("tmp");

    auto wall_start = Clock::now();

    // =====================================================
    // Phase 1a: Generate X-sorted runs
    // =====================================================
    g_stats = {};
    auto phase1a_start = Clock::now();

    size_t ram_budget = USABLE_RAM_BYTES;
    std::vector<CachedRun> runs;
    generate_runs_cached(input, SORT_CHUNK_POINTS, runs, ram_budget);

    TimingStats gen_stats = g_stats;
    gen_stats.wall_ms = Ms(Clock::now() - phase1a_start).count();
    gen_stats.print("Phase 1a — Generate X-sorted runs");

    size_t cached = 0, on_disk = 0;
    for (auto& r : runs) {
        if (!r.data.empty()) cached++;
        else on_disk++;
    }
    std::cout << "  Runs: " << runs.size() << " total, "
              << cached << " cached in RAM, "
              << on_disk << " on disk\n";

    // =====================================================
    // Phase 1b: Merge X-sorted runs
    // =====================================================
    g_stats = {};
    auto phase1b_start = Clock::now();

    const std::string sorted_x = "tmp/sorted_x.bin";
    std::vector<Point> sorted_mem;
    bool data_in_memory = false;

    if (runs.size() <= 1) {
        if (runs.size() == 1) {
            if (!runs[0].data.empty()) {
                sorted_mem = std::move(runs[0].data);
                data_in_memory = true;
            } else {
                std::filesystem::rename(runs[0].path, sorted_x);
            }
        }
        runs.clear();
    } else {
        kway_merge(runs, sorted_x, sorted_mem, data_in_memory,
                   total_points, true);
        runs.clear();
    }

    TimingStats merge_stats = g_stats;
    merge_stats.wall_ms = Ms(Clock::now() - phase1b_start).count();
    merge_stats.print("Phase 1b — Merge X-sorted runs");

    auto phase1_end = Clock::now();
    std::cout << "  Merged data: "
              << (data_in_memory ? "in RAM" : "on disk") << "\n";
    std::cout << "Phase 1 wall time: "
              << Ms(phase1_end - phase1a_start).count() << " ms\n";

    // =====================================================
    // Phase 2: STR Y-sort slices + leaf MBRs + write output
    //
    // Dual-thread: CPU sort ‖ serialized I/O
    // =====================================================
    g_stats = {};
    auto phase2_start = Clock::now();

    size_t num_nodes   = precompute_num_nodes(total_points,
                                              RTREE_LEAF_CAPACITY,
                                              RTREE_INTERNAL_CAPACITY);
    size_t num_leaves  = (total_points + RTREE_LEAF_CAPACITY - 1)
                       / RTREE_LEAF_CAPACITY;
    size_t points_offset = sizeof(RTreeHeader)
                         + num_nodes * sizeof(RTreeNode);

    std::cout << "  Pre-computed: " << num_nodes << " total nodes, "
              << num_leaves << " leaves, points offset = "
              << points_offset / (1024.0*1024.0) << " MB\n";

    std::vector<RTreeNode> nodes;
    nodes.reserve(num_nodes);

    size_t slice_bytes = slice * sizeof(Point);

    // 2 host buffers for dual-thread Y-sort pipeline
    std::vector<Point> sbuf0(slice);
    std::vector<Point> sbuf1(slice);
    Point* sbufs[2] = { sbuf0.data(), sbuf1.data() };

    size_t points_written = 0;

    // Source state for reading X-sorted data
    size_t mem_offset = 0;
    std::ifstream in_file;
    if (!data_in_memory)
        in_file.open(sorted_x, std::ios::binary);

    // Abstracted slice reader (memory or file)
    auto read_slice = [&](Point* dst, size_t max_pts)
                          -> std::pair<size_t, double> {
        auto t0 = Clock::now();
        size_t n = 0;
        if (data_in_memory) {
            size_t avail = total_points - mem_offset;
            n = std::min(max_pts, avail);
            if (n > 0) {
                memcpy(dst, sorted_mem.data() + mem_offset,
                       n * sizeof(Point));
                mem_offset += n;
            }
        } else {
            in_file.read(reinterpret_cast<char*>(dst),
                         max_pts * sizeof(Point));
            n = in_file.gcount() / sizeof(Point);
        }
        double ms = Ms(Clock::now() - t0).count();
        return {n, ms};
    };

    // Open output file — reserve space for header + nodes
    std::ofstream out_file(output, std::ios::binary);
    {
        constexpr size_t ZBUF_SZ = 1024 * 1024;
        std::vector<char> zbuf(std::min(points_offset, ZBUF_SZ), 0);
        size_t remaining = points_offset;
        while (remaining > 0) {
            size_t w = std::min(remaining, ZBUF_SZ);
            out_file.write(zbuf.data(), w);
            remaining -= w;
        }
    }

    // Pre-read first slice
    auto [n0, read0_ms] = read_slice(sbufs[0], slice);
    g_stats.disk_read_ms += read0_ms;
    g_stats.bytes_read   += n0 * sizeof(Point);

    int sort_buf = 0, io_buf = 1;
    size_t sort_n = n0;
    bool src_done = (n0 == 0);

    // Pending write state
    Point* pw_data = nullptr;
    size_t pw_n    = 0;

    while (sort_n > 0) {
        // Snapshot for I/O thread
        Point* wd       = pw_data;
        size_t wn       = pw_n;
        bool   do_read  = !src_done;
        Point* read_dst = sbufs[io_buf];

        // --- I/O thread: write previous slice, then read next ---
        std::future<std::tuple<size_t, double, double>> io_fut;
        bool do_io = (wd != nullptr) || do_read;

        if (do_io) {
            io_fut = std::async(std::launch::async,
                [wd, wn, &out_file, &read_slice, read_dst, slice,
                 do_read]()
                    -> std::tuple<size_t, double, double> {
                    double write_ms = 0, read_ms = 0;
                    size_t nr = 0;

                    if (wd) {
                        auto tw0 = Clock::now();
                        out_file.write(reinterpret_cast<char*>(wd),
                                       wn * sizeof(Point));
                        write_ms = Ms(Clock::now() - tw0).count();
                    }

                    if (do_read) {
                        auto [n, ms] = read_slice(read_dst, slice);
                        nr = n;
                        read_ms = ms;
                    }

                    return {nr, write_ms, read_ms};
                });
        }

        // --- Main thread: CPU sort current slice by Y ---
        cpu_sort(sbufs[sort_buf], sort_n, false);

        // Compute leaf MBRs (data is cache-hot after sort)
        {
            size_t cap = RTREE_LEAF_CAPACITY;
            size_t nleaves = (sort_n + cap - 1) / cap;
            for (size_t li = 0; li < nleaves; li++) {
                size_t start = li * cap;
                size_t cnt   = std::min(cap, sort_n - start);

                MBR m{INFINITY, INFINITY, -INFINITY, -INFINITY};
                for (size_t j = 0; j < cnt; j++) {
                    const Point& p = sbufs[sort_buf][start + j];
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

        // Mark this buffer as pending write
        pw_data = sbufs[sort_buf];
        pw_n    = sort_n;

        // Collect I/O results
        size_t next_n = 0;
        if (do_io) {
            auto [nr, wms, rms] = io_fut.get();
            g_stats.disk_write_ms += wms;
            if (wd) g_stats.bytes_written += wn * sizeof(Point);
            g_stats.disk_read_ms  += rms;
            g_stats.bytes_read    += nr * sizeof(Point);
            next_n = nr;
            if (nr == 0 && do_read) src_done = true;
        }

        // Swap buffers
        std::swap(sort_buf, io_buf);
        sort_n = next_n;
    }

    // Flush last sorted slice
    if (pw_data) {
        auto tw0 = Clock::now();
        out_file.write(reinterpret_cast<char*>(pw_data),
                       pw_n * sizeof(Point));
        auto tw1 = Clock::now();
        g_stats.disk_write_ms += Ms(tw1 - tw0).count();
        g_stats.bytes_written += pw_n * sizeof(Point);
    }

    if (in_file.is_open()) in_file.close();
    out_file.close();

    // Free X-sorted data
    sorted_mem.clear();
    sorted_mem.shrink_to_fit();
    if (!data_in_memory)
        std::filesystem::remove(sorted_x);

    TimingStats tile_stats = g_stats;
    tile_stats.wall_ms = Ms(Clock::now() - phase2_start).count();
    tile_stats.print("Phase 2 — STR Y-sort + leaf MBRs + write");

    // =====================================================
    // Phase 3: Build internal R-tree levels + write header
    // =====================================================
    auto t_tree0 = Clock::now();
    uint32_t height;
    assert(nodes.size() == num_leaves);
    build_internal_levels(nodes, num_leaves, RTREE_INTERNAL_CAPACITY,
                          height);
    assert(nodes.size() == num_nodes);
    auto t_tree1 = Clock::now();
    double tree_ms = Ms(t_tree1 - t_tree0).count();

    auto t_hdr0 = Clock::now();
    {
        RTreeHeader hdr{};
        hdr.magic             = 0x52545245;  // "RTRE"
        hdr.leaf_capacity     = (uint32_t)RTREE_LEAF_CAPACITY;
        hdr.height            = height;
        hdr.internal_capacity = (uint32_t)RTREE_INTERNAL_CAPACITY;
        hdr.num_points        = total_points;
        hdr.num_nodes         = nodes.size();

        std::fstream fout(output, std::ios::binary
                                | std::ios::in | std::ios::out);
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

    TimingStats phase3_stats;
    phase3_stats.tree_ms = tree_ms;
    phase3_stats.hdr_ms  = hdr_ms;
    phase3_stats.wall_ms = tree_ms + hdr_ms;  // single-threaded, no overlap
    phase3_stats.print("Phase 3 — Build internal levels + write header");

    auto wall_end = Clock::now();

    // =====================================================
    // Grand total — folds in every phase (incl. Phase 3) so the WALL line
    // reconciles with the total wall-clock time below.
    // =====================================================
    TimingStats total_stats;
    total_stats += gen_stats;
    total_stats += merge_stats;
    total_stats += tile_stats;
    total_stats += phase3_stats;
    total_stats.wall_ms = Ms(wall_end - wall_start).count();
    total_stats.print("GRAND TOTAL (I/O + CPU + tree)");

    size_t out_bytes = sizeof(RTreeHeader)
                     + nodes.size() * sizeof(RTreeNode)
                     + total_points * sizeof(Point);

    std::cout << std::fixed << std::setprecision(2)
              << "\nTotal wall-clock time: "
              << Ms(wall_end - wall_start).count() / 1000.0
              << " s\n"
              << "STR R-Tree complete (CPU-only)." << std::endl;
}

// =========================================================
// DISPATCHER
// =========================================================

void external_str(const std::string& input,
                  const std::string& output) {

    size_t file_size = std::filesystem::file_size(input);
    size_t total_points = file_size / sizeof(Point);

    std::cout << "=== CPU-only STR R-Tree Builder ===" << std::endl;
    std::cout << "Total points:       " << total_points << std::endl;
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

    external_str_build(input, output, total_points);
}

// =========================================================
// MAIN
// =========================================================

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "Usage: ./external_str_cpu input.bin output.bin"
                  << std::endl;
        return 0;
    }

    external_str(argv[1], argv[2]);

    return 0;
}
