// str_rtree_cuda.cu
// STR (Sort-Tile-Recursive) R-Tree bulk-loading over RECTANGLES, on the GPU,
// for datasets larger than VRAM and larger than RAM.
//
// Textbook node layout: a node stores its CHILDREN's MBRs, so one page fetch
// yields every pruning decision for that node's fan.  See rect_rtree_format.h.
//
// The MBR of a node is computed exactly once, when the node is packed, and is
// handed to the NEXT level as an in-memory Entry.  The raw data is therefore
// read exactly once no matter how tall the tree gets: level L+1 is built from
// level L's in-RAM entry array, never from the rectangles on disk.
//
// Pipeline:
//   Phase A  GPU sort by centroid-X into runs (RAM-cached or spilled)
//   Phase B  single-pass k-way heap merge of the runs
//   Phase C  per STR slice: GPU sort by centroid-Y, pack leaf pages, emit
//            one in-memory Entry per leaf
//   Phase D  repeat STR over the in-memory level, packing internal pages,
//            until one node remains
//   Phase E  write the file header (page 0)
//
// Compile: nvcc -O3 -std=c++17 -arch=sm_86 str_rtree_cuda.cu -o str_rtree

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <future>
#include <tuple>
#include <thread>

#include "rect_rtree_format.h"
#include "kway_merge.h"

// =========================================================
// GPU chunk sizing
// =========================================================
//
// A radix sort_by_key needs double buffers for BOTH keys and values, so the
// device must hold roughly 2 x (Entry + key) per element.
constexpr size_t GPU_BYTES_PER_ELEM = 2 * (sizeof(Entry) + sizeof(uint32_t)); // 56
constexpr size_t GPU_CHUNK_ELEMS    = USABLE_GPU_BYTES / GPU_BYTES_PER_ELEM;

// The chunk is bounded by whichever is smaller: what the GPU can sort, or what
// we are willing to pin on the host.  Pinning is linear in size and transfer
// bandwidth is flat, so the pinned bound is usually the binding one.
constexpr size_t SORT_CHUNK_ELEMS =
    (GPU_CHUNK_ELEMS < MAX_PINNED_CHUNK_BYTES / sizeof(Entry))
        ? GPU_CHUNK_ELEMS
        : MAX_PINNED_CHUNK_BYTES / sizeof(Entry);

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// =========================================================
// Timing
// =========================================================
//
// Threading contract: the async I/O worker NEVER writes g_stats.  It returns
// its timings through a std::future and the MAIN thread folds them in.  No
// field is touched by two threads, so no lock is needed.  Preserve this.
struct TimingStats {
    double disk_read_ms = 0, disk_write_ms = 0;
    double h2d_ms = 0, d2h_ms = 0, gpu_compute_ms = 0, gpu_alloc_ms = 0;
    double cpu_sort_ms = 0, cpu_merge_ms = 0, pack_ms = 0;
    double setup_ms = 0;      // pinned/host buffer allocation and release
    double wall_ms = 0;
    size_t bytes_read = 0, bytes_written = 0, bytes_h2d = 0, bytes_d2h = 0;

    double component_sum_ms() const {
        return disk_read_ms + disk_write_ms + gpu_alloc_ms + h2d_ms
             + gpu_compute_ms + d2h_ms + cpu_sort_ms + cpu_merge_ms + pack_ms
             + setup_ms;
    }

    void print(const char* phase) const {
        auto bw = [](size_t b, double ms) {
            return ms > 0 ? (b / (1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
        };
        std::cout << "\n===== Timing: " << phase << " =====\n"
                  << std::fixed << std::setprecision(2)
                  << "  Disk read:     " << disk_read_ms  << " ms  ("
                  << bw(bytes_read, disk_read_ms) << " MB/s, "
                  << bytes_read / (1024.0 * 1024.0) << " MB)\n"
                  << "  Disk write:    " << disk_write_ms << " ms  ("
                  << bw(bytes_written, disk_write_ms) << " MB/s, "
                  << bytes_written / (1024.0 * 1024.0) << " MB)\n";
        if (gpu_alloc_ms > 0 || h2d_ms > 0)
            std::cout << "  GPU alloc:     " << gpu_alloc_ms << " ms\n"
                      << "  H2D transfer:  " << h2d_ms << " ms  ("
                      << bw(bytes_h2d, h2d_ms) << " MB/s, "
                      << bytes_h2d / (1024.0 * 1024.0) << " MB)\n"
                      << "  GPU sort:      " << gpu_compute_ms << " ms\n"
                      << "  D2H transfer:  " << d2h_ms << " ms  ("
                      << bw(bytes_d2h, d2h_ms) << " MB/s, "
                      << bytes_d2h / (1024.0 * 1024.0) << " MB)\n";
        if (cpu_sort_ms  > 0) std::cout << "  CPU sort:      " << cpu_sort_ms  << " ms\n";
        if (cpu_merge_ms > 0) std::cout << "  CPU merge:     " << cpu_merge_ms << " ms\n";
        if (pack_ms      > 0) std::cout << "  Pack + MBR:    " << pack_ms      << " ms\n";
        if (setup_ms     > 0) std::cout << "  Buffer/cache:  " << setup_ms     << " ms  "
                                        << "(pinned alloc ~0.53 ms/MB + RAM-cache copies)\n";

        double c = component_sum_ms();
        std::cout << "  COMPONENT SUM: " << c << " ms  (serialized cost)\n";
        if (wall_ms > 0) {
            std::cout << "  WALL:          " << wall_ms << " ms  (actual elapsed)\n";
            // Report in BOTH directions.  Only showing the overlap saving hides
            // the opposite case -- work that happens but is not instrumented --
            // which is exactly how several seconds of pinned-buffer allocation
            // stayed invisible until the wall clock was reconciled by hand.
            if (c > wall_ms)
                std::cout << "  OVERLAP SAVED: " << (c - wall_ms)
                          << " ms  (compute || I/O concurrency)\n";
            else if (wall_ms - c > 1.0)
                std::cout << "  UNTRACKED:     " << (wall_ms - c)
                          << " ms  (work inside this phase that nothing times)\n";
        }
    }

    TimingStats& operator+=(const TimingStats& o) {
        disk_read_ms += o.disk_read_ms;   disk_write_ms += o.disk_write_ms;
        h2d_ms += o.h2d_ms;               d2h_ms += o.d2h_ms;
        gpu_compute_ms += o.gpu_compute_ms; gpu_alloc_ms += o.gpu_alloc_ms;
        cpu_sort_ms += o.cpu_sort_ms;     cpu_merge_ms += o.cpu_merge_ms;
        pack_ms += o.pack_ms;             setup_ms += o.setup_ms;
        bytes_read += o.bytes_read;       bytes_written += o.bytes_written;
        bytes_h2d += o.bytes_h2d;         bytes_d2h += o.bytes_d2h;
        return *this;
    }
};

static TimingStats g_stats;

// Threads used by the in-RAM k-way merge (Phase B). The merge is pure CPU
// work with no I/O to hide behind, so idle cores are free speedup there.
static int g_merge_threads = 1;

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": "
                  << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// =========================================================
// Sort key: total-order-preserving float -> uint32
// =========================================================
//
// Sorting a 24-byte struct with a custom comparator makes Thrust fall back to
// CUB's *comparison* merge sort (see can_use_primitive_sort: it demands an
// arithmetic key AND thrust::less/greater).  A merge sort makes ~log2(n/block)
// passes over the data; a radix sort makes 4.  In a workload that is purely
// bandwidth-bound, passes over the data ARE the running time.
//
// So the key is extracted into a separate uint32 array and sorted with
// sort_by_key and the default comparator, which reaches DeviceRadixSort.
// The transform below is the standard monotone map from IEEE-754 float order
// to unsigned integer order, valid for negatives, zeros and positives alike.
__host__ __device__ inline uint32_t order_float(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void k_make_keys(const Entry* __restrict__ src,
                            uint32_t* __restrict__ keys,
                            size_t n, int by_x) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Rect r = src[i].mbr;
    keys[i] = order_float(by_x ? (r.min_x + r.max_x) : (r.min_y + r.max_y));
}

// =========================================================
// Reusable device state
// =========================================================

static Entry*      g_d_entries = nullptr;
static uint32_t*   g_d_keys    = nullptr;
static size_t      g_d_cap     = 0;      // capacity in ELEMENTS
static cudaEvent_t g_ev0, g_ev1, g_ev2, g_ev3;
static bool        g_events_ready = false;

static void ensure_events() {
    if (g_events_ready) return;
    cudaEventCreate(&g_ev0); cudaEventCreate(&g_ev1);
    cudaEventCreate(&g_ev2); cudaEventCreate(&g_ev3);
    g_events_ready = true;
}

static void ensure_device_capacity(size_t n) {
    if (n <= g_d_cap) return;
    cudaFree(g_d_entries);
    cudaFree(g_d_keys);
    cuda_check(cudaMalloc(&g_d_entries, n * sizeof(Entry)), "cudaMalloc entries");
    cuda_check(cudaMalloc(&g_d_keys,    n * sizeof(uint32_t)), "cudaMalloc keys");
    g_d_cap = n;
}

// One-time CUDA context creation + first allocation, so the init cost is
// reported separately instead of inflating the first sort.
static double gpu_warmup(size_t elems) {
    auto t0 = Clock::now();
    cudaFree(0);
    ensure_events();
    ensure_device_capacity(elems);
    cudaDeviceSynchronize();
    return Ms(Clock::now() - t0).count();
}

static void gpu_cleanup() {
    if (g_d_entries) { cudaFree(g_d_entries); g_d_entries = nullptr; }
    if (g_d_keys)    { cudaFree(g_d_keys);    g_d_keys = nullptr; }
    g_d_cap = 0;
    if (g_events_ready) {
        cudaEventDestroy(g_ev0); cudaEventDestroy(g_ev1);
        cudaEventDestroy(g_ev2); cudaEventDestroy(g_ev3);
        g_events_ready = false;
    }
}

// Sort `n` entries in host memory by centroid X or Y, on the GPU.
static void gpu_sort_entries(Entry* host, size_t n, bool by_x) {
    if (n < 2) return;

    auto t0 = Clock::now();
    ensure_device_capacity(n);
    ensure_events();
    cudaDeviceSynchronize();
    g_stats.gpu_alloc_ms += Ms(Clock::now() - t0).count();

    size_t bytes = n * sizeof(Entry);

    cudaEventRecord(g_ev0);
    cudaMemcpyAsync(g_d_entries, host, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(g_ev1);

    int block = 256;
    size_t grid = (n + block - 1) / block;
    k_make_keys<<<(unsigned)grid, block>>>(g_d_entries, g_d_keys, n, by_x ? 1 : 0);

    // Default comparator on a uint32 key -> CUB DeviceRadixSort, not merge sort.
    thrust::sort_by_key(thrust::device,
                        thrust::device_pointer_cast(g_d_keys),
                        thrust::device_pointer_cast(g_d_keys + n),
                        thrust::device_pointer_cast(g_d_entries));
    cudaEventRecord(g_ev2);

    cudaMemcpyAsync(host, g_d_entries, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(g_ev3);
    cudaEventSynchronize(g_ev3);
    cuda_check(cudaGetLastError(), "gpu_sort_entries");

    float h2d = 0, sort = 0, d2h = 0;
    cudaEventElapsedTime(&h2d,  g_ev0, g_ev1);
    cudaEventElapsedTime(&sort, g_ev1, g_ev2);
    cudaEventElapsedTime(&d2h,  g_ev2, g_ev3);

    g_stats.h2d_ms += h2d;  g_stats.bytes_h2d += bytes;
    g_stats.gpu_compute_ms += sort;
    g_stats.d2h_ms += d2h;  g_stats.bytes_d2h += bytes;
}

// Sort dispatch: the GPU is only worth its PCIe round trip above a threshold.
static void sort_entries(Entry* data, size_t n, bool by_x) {
    if (n < 2) return;
    if (n >= GPU_SORT_MIN_ELEMS && n <= SORT_CHUNK_ELEMS) {
        gpu_sort_entries(data, n, by_x);
        return;
    }
    auto t0 = Clock::now();
    if (by_x)
        std::sort(data, data + n, [](const Entry& a, const Entry& b) {
            return centroid_key_x(a.mbr) < centroid_key_x(b.mbr); });
    else
        std::sort(data, data + n, [](const Entry& a, const Entry& b) {
            return centroid_key_y(a.mbr) < centroid_key_y(b.mbr); });
    g_stats.cpu_sort_ms += Ms(Clock::now() - t0).count();
}

// =========================================================
// Cached run: a sorted run that lives EITHER in RAM or on disk.
// =========================================================

struct CachedRun {
    EntryVec data;   // non-empty -> cached in RAM
    std::string        path;   // non-empty -> spilled to disk
    size_t             count = 0;
};

// Unified reader, so the merge loop is written once regardless of where a run
// physically lives.
struct RunReader {
    const Entry* mem = nullptr;
    size_t mem_n = 0, mem_pos = 0;

    std::ifstream file;
    std::vector<Entry> buf;
    size_t buf_pos = 0, buf_end = 0;
    std::string path;

    bool is_mem = false, at_eof = false;

    void open(const CachedRun& r, size_t buf_elems) {
        if (!r.data.empty()) {
            is_mem = true; mem = r.data.data(); mem_n = r.count; mem_pos = 0;
            at_eof = (mem_n == 0);
        } else {
            is_mem = false; path = r.path;
            buf.resize(buf_elems);
            file.open(r.path, std::ios::binary);
            refill();
        }
    }
    void refill() {
        if (at_eof || is_mem) return;
        auto t0 = Clock::now();
        file.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(Entry));
        buf_end = file.gcount() / sizeof(Entry);
        buf_pos = 0;
        g_stats.disk_read_ms += Ms(Clock::now() - t0).count();
        g_stats.bytes_read   += buf_end * sizeof(Entry);
        if (buf_end == 0) at_eof = true;
    }
    bool eof() const { return at_eof; }
    const Entry& current() const { return is_mem ? mem[mem_pos] : buf[buf_pos]; }
    bool advance() {
        if (is_mem) { if (++mem_pos >= mem_n) { at_eof = true; return false; } return true; }
        if (++buf_pos >= buf_end) { refill(); if (at_eof) return false; }
        return true;
    }
    void close_and_delete() {
        if (!is_mem) {
            file.close();
            if (!path.empty()) std::filesystem::remove(path);
        }
    }
};

// =========================================================
// Phase A: GPU sort by centroid-X into runs.
//
// Main thread: GPU sort.
// I/O thread:  write-previous THEN read-next, serialized.
//
// The I/O is deliberately single-threaded and serialized: two concurrent
// streams on one SSD interleave at the device queue and turn two sequential
// patterns into one random one.  Overlap only pays across INDEPENDENT devices.
// =========================================================

static void generate_runs(const std::string& input,
                          size_t total_elems,
                          std::vector<CachedRun>& runs,
                          size_t& ram_budget) {
    size_t chunk_elems = std::min(SORT_CHUNK_ELEMS, std::max<size_t>(total_elems, 1));
    size_t chunk_bytes = chunk_elems * sizeof(Entry);

    Entry* bufs[2];
    auto ts0 = Clock::now();
    cuda_check(cudaMallocHost(&bufs[0], chunk_bytes), "cudaMallocHost A0");
    cuda_check(cudaMallocHost(&bufs[1], chunk_bytes), "cudaMallocHost A1");
    g_stats.setup_ms += Ms(Clock::now() - ts0).count();

    std::ifstream in(input, std::ios::binary);

    auto t0 = Clock::now();
    in.read(reinterpret_cast<char*>(bufs[0]), chunk_bytes);
    size_t n0 = in.gcount() / sizeof(Entry);
    g_stats.disk_read_ms += Ms(Clock::now() - t0).count();
    g_stats.bytes_read   += n0 * sizeof(Entry);

    if (n0 == 0) { cudaFreeHost(bufs[0]); cudaFreeHost(bufs[1]); return; }

    int sort_buf = 0, io_buf = 1;
    size_t sort_n = n0;
    bool input_done = false;

    Entry*      pend_data = nullptr;
    size_t      pend_n    = 0;
    std::string pend_path;

    while (sort_n > 0) {
        Entry*      wd = pend_data;
        size_t      wn = pend_n;
        std::string wp = pend_path;
        bool        do_read  = !input_done;
        Entry*      read_dst = bufs[io_buf];

        std::future<std::tuple<size_t, double, double>> io;
        bool do_io = (wd != nullptr) || do_read;
        if (do_io) {
            io = std::async(std::launch::async,
                [wd, wn, wp, read_dst, chunk_bytes, &in, do_read]() {
                    double wms = 0, rms = 0; size_t nr = 0;
                    if (wd) {
                        auto a = Clock::now();
                        std::ofstream f(wp, std::ios::binary);
                        f.write(reinterpret_cast<const char*>(wd), wn * sizeof(Entry));
                        f.close();
                        wms = Ms(Clock::now() - a).count();
                    }
                    if (do_read) {
                        auto a = Clock::now();
                        in.read(reinterpret_cast<char*>(read_dst), chunk_bytes);
                        nr = in.gcount() / sizeof(Entry);
                        rms = Ms(Clock::now() - a).count();
                    }
                    return std::make_tuple(nr, wms, rms);
                });
        }

        gpu_sort_entries(bufs[sort_buf], sort_n, /*by_x=*/true);

        size_t run_bytes = sort_n * sizeof(Entry);
        CachedRun run;
        run.count = sort_n;
        if (ram_budget >= run_bytes) {
            auto tc0 = Clock::now();
            run.data.assign(bufs[sort_buf], bufs[sort_buf] + sort_n);
            g_stats.setup_ms += Ms(Clock::now() - tc0).count();
            ram_budget -= run_bytes;
            pend_data = nullptr;
        } else {
            run.path  = "tmp/x_run_" + std::to_string(runs.size()) + ".bin";
            pend_data = bufs[sort_buf];
            pend_n    = sort_n;
            pend_path = run.path;
        }
        runs.push_back(std::move(run));

        size_t next_n = 0;
        if (do_io) {
            auto [nr, wms, rms] = io.get();
            g_stats.disk_write_ms += wms;
            if (wd) g_stats.bytes_written += wn * sizeof(Entry);
            g_stats.disk_read_ms += rms;
            g_stats.bytes_read   += nr * sizeof(Entry);
            next_n = nr;
            if (nr == 0 && do_read) input_done = true;
        }

        std::swap(sort_buf, io_buf);
        sort_n = next_n;
    }

    if (pend_data) {
        auto a = Clock::now();
        std::ofstream f(pend_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(pend_data), pend_n * sizeof(Entry));
        f.close();
        g_stats.disk_write_ms += Ms(Clock::now() - a).count();
        g_stats.bytes_written += pend_n * sizeof(Entry);
    }

    in.close();
    auto ts1 = Clock::now();
    cudaFreeHost(bufs[0]);
    cudaFreeHost(bufs[1]);
    g_stats.setup_ms += Ms(Clock::now() - ts1).count();
}

// =========================================================
// Phase B: single-pass k-way heap merge.
//
// Pairwise merging costs O(N log k) BYTES of disk traffic; a k-way heap merge
// costs O(N) — one read, one write, whatever k is.  With M memory and block
// size B, this is the two-pass optimum of the external-memory model, and the
// RAM cache collapses it to ONE pass whenever every run stayed resident.
// =========================================================

static void kway_merge(std::vector<CachedRun>& runs,
                       const std::string& disk_out,
                       EntryVec& mem_out,
                       bool& all_in_memory,
                       size_t total_elems,
                       bool by_x) {
    size_t k = runs.size();
    if (k == 0) { all_in_memory = true; return; }

    bool all_cached = true;
    for (auto& r : runs) if (r.data.empty()) { all_cached = false; break; }
    all_in_memory = all_cached;

    constexpr size_t MIN_BUF = 4096;
    constexpr size_t MAX_BUF = 8u << 20;
    size_t buf_elems = MERGE_BUFFER_BYTES / ((k + 1) * sizeof(Entry));
    buf_elems = std::min(std::max(buf_elems, MIN_BUF), MAX_BUF);
    // Never buffer more than a run can possibly hold: with a large budget and
    // a small dataset the formula above would otherwise reserve gigabytes of
    // read buffers for runs of a few megabytes.
    buf_elems = std::min(buf_elems, std::max(ceil_div(total_elems, k), MIN_BUF));

    std::cout << "  K-way merge: " << k << " runs"
              << (all_cached ? " (all in RAM)" : " (some spilled)")
              << ", " << buf_elems << " entries/buffer\n";

    std::vector<RunReader> rd(k);
    for (size_t i = 0; i < k; i++) rd[i].open(runs[i], buf_elems);

    if (all_in_memory) {
        // Every run is RAM-resident, so this is pure CPU work with no I/O to
        // hide behind: partition the output into disjoint key ranges and merge
        // them in parallel.  ~10x the binary heap, and flat in k.
        auto t0 = Clock::now();
        mem_out.resize(total_elems);
        std::vector<RunSpan> spans;
        spans.reserve(runs.size());
        for (auto& r : runs) spans.push_back({ r.data.data(), r.count });
        kway_merge_ram(spans, mem_out.data(), total_elems, by_x, g_merge_threads);
        g_stats.cpu_merge_ms += Ms(Clock::now() - t0).count();
    } else {
        auto t0 = Clock::now();
        double disk_before = g_stats.disk_write_ms;
        std::vector<Entry> out(buf_elems);
        size_t used = 0;
        std::ofstream f(disk_out, std::ios::binary);

        auto flush = [&]() {
            if (!used) return;
            auto a = Clock::now();
            f.write(reinterpret_cast<char*>(out.data()), used * sizeof(Entry));
            g_stats.disk_write_ms += Ms(Clock::now() - a).count();
            g_stats.bytes_written += used * sizeof(Entry);
            used = 0;
        };

        // Spilled runs are read-once streams, so a partitioned merge is not
        // available (it would need binary search inside each file).  A
        // streaming loser tree does the same I/O with fewer, more predictable
        // comparisons — measured 1.2-1.4x faster than the binary heap here.
        loser_merge_stream(rd, by_x, [&](const Entry& e) {
            out[used++] = e;
            if (used >= buf_elems) flush();
        });
        for (auto& r : rd) r.close_and_delete();
        flush();
        f.close();
        double elapsed = Ms(Clock::now() - t0).count();
        g_stats.cpu_merge_ms += elapsed - (g_stats.disk_write_ms - disk_before);
    }

    for (auto& r : runs) { r.data.clear(); r.data.shrink_to_fit(); }
}

// =========================================================
// Page writer — sequential, page-aligned appends.
// Page 0 is reserved for the file header, so node pages start at page 1 and
// every node is page-aligned: expanding a node is exactly one page fault.
// =========================================================

struct PageWriter {
    std::ofstream out;
    std::vector<char> buf;
    size_t used = 0;
    uint64_t next_page = 0;
    size_t bytes_written = 0;

    void open(const std::string& path) {
        out.open(path, std::ios::binary | std::ios::trunc);
        buf.resize(PAGE_WRITE_BUFFER_BYTES);
        // reserve page 0 for the header
        std::vector<char> zero(PAGE_BYTES, 0);
        out.write(zero.data(), PAGE_BYTES);
        bytes_written += PAGE_BYTES;
        next_page = 1;
    }

    // Append one packed node; returns its page id. Caller times this.
    uint64_t append(const Entry* e, uint32_t count, uint32_t is_leaf, double& ms) {
        if (used + PAGE_BYTES > buf.size()) flush(ms);
        pack_page(buf.data() + used, e, count, is_leaf);
        used += PAGE_BYTES;
        return next_page++;
    }

    // Bulk-write already-packed pages (used by the leaf phase, from the I/O
    // thread). Returns elapsed ms rather than touching the global stats.
    double write_raw(const char* p, size_t n) {
        auto t0 = Clock::now();
        out.write(p, n);
        bytes_written += n;
        return Ms(Clock::now() - t0).count();
    }

    void flush(double& ms) {
        if (!used) return;
        auto t0 = Clock::now();
        out.write(buf.data(), used);
        bytes_written += used;
        ms += Ms(Clock::now() - t0).count();
        used = 0;
    }

    void close(double& ms) { flush(ms); out.close(); }
};

// =========================================================
// Pack a sorted, STR-tiled range into nodes.
//
// Groups of `cap` consecutive entries become one node.  Slice widths are
// aligned to `cap` upstream, so a node never straddles a slice boundary.
// Each packed node yields ONE Entry { node MBR, node page } for the parent
// level — the MBR is computed exactly once, here, and lives in RAM until the
// parent consumes it.  Nothing is ever re-read to recover it.
// =========================================================

static void pack_nodes(const Entry* src, size_t n, size_t cap,
                       uint32_t is_leaf, PageWriter& pw,
                       std::vector<Entry>& parent_out, double& write_ms) {
    size_t groups = ceil_div(n, cap);
    for (size_t g = 0; g < groups; g++) {
        size_t s = g * cap;
        size_t c = std::min(cap, n - s);
        Rect m = mbr_of(src + s, c);
        uint64_t page = pw.append(src + s, (uint32_t)c, is_leaf, write_ms);
        parent_out.push_back(Entry{m, page});
    }
}

// =========================================================
// One STR pass over an in-RAM level: sort by centroid X, cut into vertical
// slices aligned to `cap`, sort each slice by centroid Y, then pack.
//
// Re-tiling at EVERY level is what keeps MBRs square.  Grouping consecutive
// nodes without re-sorting walks a tall narrow column of the previous level's
// ordering and produces long thin MBRs, which is what actually costs range
// queries — dead area becomes false-positive page fetches.
// =========================================================

static std::vector<Entry> str_pass(std::vector<Entry>& level, size_t cap,
                                   uint32_t is_leaf, PageWriter& pw,
                                   double& write_ms) {
    std::vector<Entry> parent;
    size_t n = level.size();
    if (n == 0) return parent;

    if (n <= cap) {                       // single node: this is the root
        parent.reserve(1);
        auto t0 = Clock::now();
        pack_nodes(level.data(), n, cap, is_leaf, pw, parent, write_ms);
        g_stats.pack_ms += Ms(Clock::now() - t0).count();
        return parent;
    }

    sort_entries(level.data(), n, /*by_x=*/true);

    size_t slice = compute_slice_elems(n, cap, SORT_CHUNK_ELEMS);
    parent.reserve(ceil_div(n, cap));

    for (size_t off = 0; off < n; off += slice) {
        size_t cnt = std::min(slice, n - off);
        sort_entries(level.data() + off, cnt, /*by_x=*/false);
        auto t0 = Clock::now();
        pack_nodes(level.data() + off, cnt, cap, is_leaf, pw, parent, write_ms);
        g_stats.pack_ms += Ms(Clock::now() - t0).count();
    }
    return parent;
}

// =========================================================
// Builder
// =========================================================

struct BuildOptions {
    float fill_leaf     = DEFAULT_FILL_LEAF;
    float fill_internal = DEFAULT_FILL_INTERNAL;
    int   merge_threads = 0;      // 0 = hardware_concurrency
};

static void build(const std::string& input, const std::string& output,
                  const BuildOptions& opt) {
    size_t file_bytes  = std::filesystem::file_size(input);
    if (file_bytes % sizeof(Entry) != 0)
        std::cerr << "warning: input size is not a multiple of "
                  << sizeof(Entry) << " bytes; trailing bytes ignored\n";
    size_t total_elems = file_bytes / sizeof(Entry);
    if (total_elems == 0) { std::cerr << "empty input\n"; std::exit(1); }

    size_t leaf_cap = apply_fill(MAX_ENTRIES_PER_PAGE, opt.fill_leaf);
    size_t int_cap  = apply_fill(MAX_ENTRIES_PER_PAGE, opt.fill_internal);

    size_t exp_nodes  = precompute_num_nodes(total_elems, leaf_cap, int_cap);
    uint32_t exp_hgt  = precompute_height(total_elems, leaf_cap, int_cap);
    size_t num_leaves = ceil_div(total_elems, leaf_cap);

    std::cout << std::fixed << std::setprecision(2)
              << "Rectangles:          " << total_elems
              << "  (" << file_bytes / (1024.0*1024.0) << " MB)\n"
              << "PAGE_BYTES:          " << PAGE_BYTES << "\n"
              << "Entries/page (max):  " << MAX_ENTRIES_PER_PAGE << "\n"
              << "Fill leaf/internal:  " << opt.fill_leaf << " / " << opt.fill_internal << "\n"
              << "Effective capacity:  " << leaf_cap << " leaf, " << int_cap << " internal\n"
              << "Expected leaves:     " << num_leaves << "\n"
              << "Expected nodes:      " << exp_nodes
              << "  (" << exp_nodes * PAGE_BYTES / (1024.0*1024.0) << " MB)\n"
              << "Expected height:     " << exp_hgt << "\n"
              << "USABLE_GPU_BYTES:    " << USABLE_GPU_BYTES / (1024*1024) << " MB\n"
              << "USABLE_RAM_BYTES:    " << USABLE_RAM_BYTES / (1024*1024) << " MB\n"
              << "GPU chunk ceiling:   " << GPU_CHUNK_ELEMS << "\n"
              << "SORT_CHUNK_ELEMS:    " << SORT_CHUNK_ELEMS
              << "  (" << SORT_CHUNK_ELEMS * sizeof(Entry) / (1024.0*1024.0) << " MB)\n";

    std::filesystem::create_directories("tmp");
    auto wall0 = Clock::now();

    double init_ms = gpu_warmup(std::min(SORT_CHUNK_ELEMS, total_elems));
    std::cout << "GPU init (context + buffers): " << init_ms << " ms\n";

    // ---------------- Phase A ----------------
    g_stats = {};
    auto tA = Clock::now();
    size_t ram_budget = USABLE_RAM_BYTES;
    std::vector<CachedRun> runs;
    generate_runs(input, total_elems, runs, ram_budget);
    TimingStats sA = g_stats;
    sA.wall_ms = Ms(Clock::now() - tA).count();
    sA.print("Phase A - GPU sort by centroid-X into runs");

    size_t cached = 0;
    for (auto& r : runs) if (!r.data.empty()) cached++;
    std::cout << "  Runs: " << runs.size() << " total, " << cached
              << " cached in RAM, " << (runs.size() - cached) << " spilled\n";

    // ---------------- Phase B ----------------
    g_stats = {};
    auto tB = Clock::now();
    const std::string sorted_x = "tmp/rect_sorted_x.bin";
    EntryVec sorted_mem;
    bool in_memory = false;

    if (runs.size() <= 1) {
        if (runs.size() == 1) {
            if (!runs[0].data.empty()) { sorted_mem = std::move(runs[0].data); in_memory = true; }
            else std::filesystem::rename(runs[0].path, sorted_x);
        }
        runs.clear();
    } else {
        kway_merge(runs, sorted_x, sorted_mem, in_memory, total_elems, true);
        runs.clear();
    }
    TimingStats sB = g_stats;
    sB.wall_ms = Ms(Clock::now() - tB).count();
    sB.print("Phase B - k-way merge of X-sorted runs");
    std::cout << "  Merged data: " << (in_memory ? "in RAM" : "on disk") << "\n";

    // ---------------- Phase C: leaves ----------------
    g_stats = {};
    auto tC = Clock::now();

    PageWriter pw;
    pw.open(output);

    // Host budget: two slice buffers (entries) + two page buffers (packed nodes).
    size_t page_bytes_per_elem = ceil_div(PAGE_BYTES, leaf_cap) + 1;
    size_t per_elem = 2 * (sizeof(Entry) + page_bytes_per_elem);
    size_t host_max = PHASE_LEAF_HOST_BYTES / per_elem;
    size_t slice = compute_slice_elems(total_elems, leaf_cap,
                                       std::min(SORT_CHUNK_ELEMS, host_max));
    size_t slice_pages = ceil_div(slice, leaf_cap) + 1;

    std::cout << "  STR slice: " << slice << " entries ("
              << slice * sizeof(Entry) / (1024.0*1024.0) << " MB), "
              << slice_pages << " pages/slice buffer\n";

    auto tsC = Clock::now();
    Entry* sbuf[2];
    cuda_check(cudaMallocHost(&sbuf[0], slice * sizeof(Entry)), "cudaMallocHost C0");
    cuda_check(cudaMallocHost(&sbuf[1], slice * sizeof(Entry)), "cudaMallocHost C1");
    ByteVec pbuf0, pbuf1; pbuf0.resize(slice_pages * PAGE_BYTES); pbuf1.resize(slice_pages * PAGE_BYTES);
    g_stats.setup_ms += Ms(Clock::now() - tsC).count();
    char* pbuf[2] = { pbuf0.data(), pbuf1.data() };

    std::vector<Entry> level;             // one Entry per leaf, kept in RAM
    level.reserve(num_leaves);

    size_t mem_pos = 0;
    std::ifstream in_x;
    if (!in_memory) in_x.open(sorted_x, std::ios::binary);

    auto read_slice = [&](Entry* dst, size_t max_n) -> std::pair<size_t, double> {
        auto t0 = Clock::now();
        size_t n = 0;
        if (in_memory) {
            n = std::min(max_n, sorted_mem.size() - mem_pos);
            if (n) { memcpy(dst, sorted_mem.data() + mem_pos, n * sizeof(Entry)); mem_pos += n; }
        } else {
            in_x.read(reinterpret_cast<char*>(dst), max_n * sizeof(Entry));
            n = in_x.gcount() / sizeof(Entry);
        }
        return { n, Ms(Clock::now() - t0).count() };
    };

    auto [n0, r0] = read_slice(sbuf[0], slice);
    g_stats.disk_read_ms += r0;
    g_stats.bytes_read   += n0 * sizeof(Entry);

    int cur = 0, other = 1;
    size_t cur_n = n0;
    bool src_done = (n0 == 0);

    char*  pend_pages = nullptr;
    size_t pend_bytes = 0;

    while (cur_n > 0) {
        char*  wp = pend_pages;
        size_t wb = pend_bytes;
        bool   do_read = !src_done;
        Entry* rdst = sbuf[other];

        std::future<std::tuple<size_t, double, double>> io;
        bool do_io = (wp != nullptr) || do_read;
        if (do_io) {
            io = std::async(std::launch::async, [&, wp, wb, rdst, do_read]() {
                double wms = 0, rms = 0; size_t nr = 0;
                if (wp) wms = pw.write_raw(wp, wb);
                if (do_read) { auto pr = read_slice(rdst, slice); nr = pr.first; rms = pr.second; }
                return std::make_tuple(nr, wms, rms);
            });
        }

        // Dispatch, don't assume: an STR slice can be far smaller than a sort
        // chunk (it is ~sqrt(num_leaves) of the data), and below the threshold
        // the PCIe round trip costs more than the sort saves.
        sort_entries(sbuf[cur], cur_n, /*by_x=*/false);

        // Pack leaves while the slice is still cache-hot from the D2H copy.
        auto tp = Clock::now();
        size_t nleaves = ceil_div(cur_n, leaf_cap);
        for (size_t g = 0; g < nleaves; g++) {
            size_t s = g * leaf_cap;
            size_t c = std::min(leaf_cap, cur_n - s);
            Rect m = mbr_of(sbuf[cur] + s, c);
            pack_page(pbuf[cur] + g * PAGE_BYTES, sbuf[cur] + s, (uint32_t)c, 1);
            level.push_back(Entry{m, pw.next_page++});
        }
        g_stats.pack_ms += Ms(Clock::now() - tp).count();

        pend_pages = pbuf[cur];
        pend_bytes = nleaves * PAGE_BYTES;

        size_t next_n = 0;
        if (do_io) {
            auto [nr, wms, rms] = io.get();
            g_stats.disk_write_ms += wms;
            g_stats.bytes_written += wb;
            g_stats.disk_read_ms  += rms;
            g_stats.bytes_read    += nr * sizeof(Entry);
            next_n = nr;
            if (nr == 0 && do_read) src_done = true;
        }

        std::swap(cur, other);
        cur_n = next_n;
    }

    if (pend_pages) {
        double wms = pw.write_raw(pend_pages, pend_bytes);
        g_stats.disk_write_ms += wms;
        g_stats.bytes_written += pend_bytes;
    }

    if (in_x.is_open()) in_x.close();
    auto tsC2 = Clock::now();
    cudaFreeHost(sbuf[0]);
    cudaFreeHost(sbuf[1]);
    g_stats.setup_ms += Ms(Clock::now() - tsC2).count();
    sorted_mem.clear(); sorted_mem.shrink_to_fit();
    if (!in_memory) std::filesystem::remove(sorted_x);

    TimingStats sC = g_stats;
    sC.wall_ms = Ms(Clock::now() - tC).count();
    sC.print("Phase C - GPU sort by centroid-Y, pack leaf pages");
    std::cout << "  Leaves packed: " << level.size()
              << " (expected " << num_leaves << ")\n";

    // ---------------- Phase D: internal levels ----------------
    g_stats = {};
    auto tD = Clock::now();
    double write_ms = 0;

    uint32_t height = 1;
    uint64_t root_page = (level.size() == 1) ? level[0].child : 0;
    Rect root_mbr = level.empty() ? mbr_empty() : level[0].mbr;

    while (level.size() > 1) {
        std::vector<Entry> parent = str_pass(level, int_cap, /*is_leaf=*/0, pw, write_ms);
        height++;
        std::cout << "  Level " << height << ": " << level.size()
                  << " entries -> " << parent.size() << " nodes\n";
        level.swap(parent);
    }
    if (!level.empty()) { root_page = level[0].child; root_mbr = level[0].mbr; }

    pw.close(write_ms);
    g_stats.disk_write_ms += write_ms;
    g_stats.bytes_written += pw.bytes_written;

    TimingStats sD = g_stats;
    sD.wall_ms = Ms(Clock::now() - tD).count();
    sD.print("Phase D - build internal levels");

    // ---------------- Phase E: header ----------------
    uint64_t num_nodes = pw.next_page - 1;
    {
        FileHeader h{};
        h.magic                = RTREE_MAGIC;
        h.page_bytes           = (uint32_t)PAGE_BYTES;
        h.height               = height;
        h.max_entries_per_page = (uint32_t)MAX_ENTRIES_PER_PAGE;
        h.leaf_cap             = (uint32_t)leaf_cap;
        h.internal_cap         = (uint32_t)int_cap;
        h.fill_leaf            = opt.fill_leaf;
        h.fill_internal        = opt.fill_internal;
        h.num_rects            = total_elems;
        h.num_nodes            = num_nodes;
        h.root_page            = root_page;
        h.root_mbr             = root_mbr;

        std::fstream f(output, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(0);
        f.write(reinterpret_cast<const char*>(&h), sizeof(h));
        f.close();
    }

    auto wall1 = Clock::now();
    TimingStats total;
    total += sA; total += sB; total += sC; total += sD;
    total.setup_ms += init_ms;      // CUDA context creation is a real cost
    total.wall_ms = Ms(wall1 - wall0).count();
    total.print("GRAND TOTAL");

    std::cout << "\nR-Tree written: " << output << "\n"
              << "  Nodes:   " << num_nodes << " pages ("
              << num_nodes * PAGE_BYTES / (1024.0*1024.0) << " MB)"
              << "   expected " << exp_nodes << "\n"
              << "  Height:  " << height << "   expected " << exp_hgt << "\n"
              << "  Root:    page " << root_page << "\n"
              << "  Root MBR: [" << root_mbr.min_x << ", " << root_mbr.min_y
              << "] x [" << root_mbr.max_x << ", " << root_mbr.max_y << "]\n"
              << "\nTotal wall-clock: " << Ms(wall1 - wall0).count() / 1000.0 << " s\n";

    if (num_nodes != exp_nodes)
        std::cerr << "WARNING: node count " << num_nodes
                  << " != precomputed " << exp_nodes << "\n";
    if (height != exp_hgt)
        std::cerr << "WARNING: height " << height
                  << " != precomputed " << exp_hgt << "\n";

    gpu_cleanup();
}

// =========================================================
// main
// =========================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout <<
        "Usage: " << argv[0] << " <input.bin> <output.bin> [options]\n\n"
        "  input.bin   Entry records { Rect(16B), uint64 id } = 24 B each\n"
        "              (produced by ./bin/gen_rects)\n\n"
        "Options:\n"
        "  --fill-leaf F       leaf fill factor      (0 < F <= 1, default "
        << DEFAULT_FILL_LEAF << ")\n"
        "  --fill-internal F   internal fill factor  (0 < F <= 1, default "
        << DEFAULT_FILL_INTERNAL << ")\n"
        "  --fill F            set both\n"
        "  --merge-threads N   threads for the in-RAM k-way merge "
        "(default: all cores)\n";
        return 1;
    }

    std::string input = argv[1], output = argv[2];
    BuildOptions opt;

    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--fill-leaf")     opt.fill_leaf = std::stof(next());
        else if (a == "--fill-internal") opt.fill_internal = std::stof(next());
        else if (a == "--fill")          opt.fill_leaf = opt.fill_internal = std::stof(next());
        else if (a == "--merge-threads") opt.merge_threads = std::stoi(next());
        else { std::cerr << "unknown option: " << a << "\n"; return 1; }
    }
    if (opt.fill_leaf <= 0 || opt.fill_leaf > 1 ||
        opt.fill_internal <= 0 || opt.fill_internal > 1) {
        std::cerr << "fill factors must satisfy 0 < F <= 1\n";
        return 1;
    }
    if (!std::filesystem::exists(input)) {
        std::cerr << "input not found: " << input << "\n";
        return 1;
    }
    g_merge_threads = opt.merge_threads > 0 ? opt.merge_threads
                                            : (int)std::thread::hardware_concurrency();
    if (g_merge_threads < 1) g_merge_threads = 1;

    build(input, output, opt);
    return 0;
}
