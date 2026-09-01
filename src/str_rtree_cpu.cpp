// str_rtree_cpu.cpp
// CPU control group for str_rtree_cuda.cu.
//
// A faithful twin: identical phase structure, identical on-disk format,
// identical cached-run / k-way-merge / double-buffered-I/O machinery.  The
// ONLY difference is where the sorting happens.  Any measured difference is
// therefore attributable to the sort and the PCIe transfers it costs, which is
// the actual research question.
//
//   --threads 1   pure single-threaded std::sort   (the honest floor)
//   --threads N   parallel merge sort over N threads
//   --threads 0   use std::thread::hardware_concurrency()
//
// Compile: g++ -O3 -std=c++17 -pthread str_rtree_cpu.cpp -o str_rtree_cpu

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
#include <thread>
#include <tuple>

#include "rect_rtree_format.h"
#include "kway_merge.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// The CPU has no VRAM ceiling, so the chunk is bounded only by the host
// staging budget.  Kept identical to the GPU build's effective chunk so the
// two produce the same number of runs and the same merge shape.
constexpr size_t SORT_CHUNK_ELEMS = MAX_PINNED_CHUNK_BYTES / sizeof(Entry);

static int g_threads = 1;
static int g_merge_threads = 1;   // follows --threads: --threads 1 is fully serial

// =========================================================
// Timing (mirrors the CUDA build; GPU fields simply stay zero)
// =========================================================

struct TimingStats {
    double disk_read_ms = 0, disk_write_ms = 0;
    double cpu_sort_ms = 0, cpu_merge_ms = 0, pack_ms = 0, setup_ms = 0;
    double wall_ms = 0;
    size_t bytes_read = 0, bytes_written = 0;

    double component_sum_ms() const {
        return disk_read_ms + disk_write_ms + cpu_sort_ms
             + cpu_merge_ms + pack_ms + setup_ms;
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
                  << bytes_written / (1024.0 * 1024.0) << " MB)\n"
                  << "  CPU sort:      " << cpu_sort_ms << " ms  ("
                  << g_threads << " thread" << (g_threads == 1 ? "" : "s") << ")\n";
        if (cpu_merge_ms > 0) std::cout << "  CPU merge:     " << cpu_merge_ms << " ms  ("
                                       << g_merge_threads << " thread"
                                       << (g_merge_threads == 1 ? "" : "s") << ")\n";
        if (pack_ms      > 0) std::cout << "  Pack + MBR:    " << pack_ms      << " ms\n";
        if (setup_ms     > 0) std::cout << "  Buffer/cache:  " << setup_ms     << " ms\n";

        double c = component_sum_ms();
        std::cout << "  COMPONENT SUM: " << c << " ms  (serialized cost)\n";
        if (wall_ms > 0) {
            std::cout << "  WALL:          " << wall_ms << " ms  (actual elapsed)\n";
            if (c > wall_ms)
                std::cout << "  OVERLAP SAVED: " << (c - wall_ms)
                          << " ms  (compute || I/O concurrency)\n";
            else if (wall_ms - c > 1.0)
                std::cout << "  UNTRACKED:     " << (wall_ms - c)
                          << " ms  (work inside this phase that nothing times)\n";
        }
    }

    TimingStats& operator+=(const TimingStats& o) {
        disk_read_ms += o.disk_read_ms; disk_write_ms += o.disk_write_ms;
        cpu_sort_ms += o.cpu_sort_ms;   cpu_merge_ms += o.cpu_merge_ms;
        pack_ms += o.pack_ms;           setup_ms += o.setup_ms;
        bytes_read += o.bytes_read;     bytes_written += o.bytes_written;
        return *this;
    }
};

static TimingStats g_stats;

// =========================================================
// Sorting
// =========================================================

template <bool ByX>
struct CentroidLess {
    bool operator()(const Entry& a, const Entry& b) const {
        return ByX ? (centroid_key_x(a.mbr) < centroid_key_x(b.mbr))
                   : (centroid_key_y(a.mbr) < centroid_key_y(b.mbr));
    }
};

// Parallel merge sort: sort T contiguous blocks in parallel, then merge them
// pairwise in log2(T) rounds, each round's merges running concurrently.
// std::inplace_merge is used so no extra full-size buffer is required (it will
// allocate a temporary internally when it can, and degrade gracefully if not).
template <class Cmp>
static void parallel_sort(Entry* data, size_t n, int threads, Cmp cmp) {
    if (threads <= 1 || n < 4096) { std::sort(data, data + n, cmp); return; }

    size_t block = (n + threads - 1) / threads;
    std::vector<size_t> bounds;
    for (size_t off = 0; off < n; off += block)
        bounds.push_back(off);
    bounds.push_back(n);

    size_t nblocks = bounds.size() - 1;

    {   // round 0: sort each block
        std::vector<std::thread> pool;
        pool.reserve(nblocks);
        for (size_t b = 0; b < nblocks; b++)
            pool.emplace_back([&, b] {
                std::sort(data + bounds[b], data + bounds[b + 1], cmp);
            });
        for (auto& t : pool) t.join();
    }

    // rounds 1..: pairwise merge, stride doubling
    for (size_t stride = 1; stride < nblocks; stride *= 2) {
        std::vector<std::thread> pool;
        for (size_t b = 0; b + stride < nblocks; b += 2 * stride) {
            size_t lo  = bounds[b];
            size_t mid = bounds[b + stride];
            size_t hi  = bounds[std::min(b + 2 * stride, nblocks)];
            pool.emplace_back([data, lo, mid, hi, cmp] {
                std::inplace_merge(data + lo, data + mid, data + hi, cmp);
            });
        }
        for (auto& t : pool) t.join();
    }
}

static void sort_entries(Entry* data, size_t n, bool by_x) {
    if (n < 2) return;
    auto t0 = Clock::now();
    if (by_x) parallel_sort(data, n, g_threads, CentroidLess<true>{});
    else      parallel_sort(data, n, g_threads, CentroidLess<false>{});
    g_stats.cpu_sort_ms += Ms(Clock::now() - t0).count();
}

// =========================================================
// Cached runs  (identical to the CUDA build)
// =========================================================

struct CachedRun {
    EntryVec data;
    std::string        path;
    size_t             count = 0;
};

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
        if (!is_mem) { file.close(); if (!path.empty()) std::filesystem::remove(path); }
    }
};

static void generate_runs(const std::string& input, size_t total_elems,
                          std::vector<CachedRun>& runs, size_t& ram_budget) {
    size_t chunk_elems = std::min(SORT_CHUNK_ELEMS, std::max<size_t>(total_elems, 1));
    size_t chunk_bytes = chunk_elems * sizeof(Entry);

    auto ts0 = Clock::now();
    EntryVec b0, b1; b0.resize(chunk_elems); b1.resize(chunk_elems);
    Entry* bufs[2] = { b0.data(), b1.data() };
    g_stats.setup_ms += Ms(Clock::now() - ts0).count();

    std::ifstream in(input, std::ios::binary);
    auto t0 = Clock::now();
    in.read(reinterpret_cast<char*>(bufs[0]), chunk_bytes);
    size_t n0 = in.gcount() / sizeof(Entry);
    g_stats.disk_read_ms += Ms(Clock::now() - t0).count();
    g_stats.bytes_read   += n0 * sizeof(Entry);
    if (n0 == 0) return;

    int sort_buf = 0, io_buf = 1;
    size_t sort_n = n0;
    bool input_done = false;
    Entry* pend_data = nullptr; size_t pend_n = 0; std::string pend_path;

    while (sort_n > 0) {
        Entry* wd = pend_data; size_t wn = pend_n; std::string wp = pend_path;
        bool do_read = !input_done; Entry* read_dst = bufs[io_buf];

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

        sort_entries(bufs[sort_buf], sort_n, true);

        size_t run_bytes = sort_n * sizeof(Entry);
        CachedRun run; run.count = sort_n;
        if (ram_budget >= run_bytes) {
            auto tc = Clock::now();
            run.data.assign(bufs[sort_buf], bufs[sort_buf] + sort_n);
            g_stats.setup_ms += Ms(Clock::now() - tc).count();
            ram_budget -= run_bytes;
            pend_data = nullptr;
        } else {
            run.path  = "tmp/x_run_cpu_" + std::to_string(runs.size()) + ".bin";
            pend_data = bufs[sort_buf]; pend_n = sort_n; pend_path = run.path;
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
}

static void kway_merge(std::vector<CachedRun>& runs, const std::string& disk_out,
                       EntryVec& mem_out, bool& all_in_memory,
                       size_t total_elems, bool by_x) {
    size_t k = runs.size();
    if (k == 0) { all_in_memory = true; return; }

    bool all_cached = true;
    for (auto& r : runs) if (r.data.empty()) { all_cached = false; break; }
    all_in_memory = all_cached;

    constexpr size_t MIN_BUF = 4096, MAX_BUF = 8u << 20;
    size_t buf_elems = MERGE_BUFFER_BYTES / ((k + 1) * sizeof(Entry));
    buf_elems = std::min(std::max(buf_elems, MIN_BUF), MAX_BUF);
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
        double before = g_stats.disk_write_ms;
        std::vector<Entry> out(buf_elems); size_t used = 0;
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
        flush(); f.close();
        g_stats.cpu_merge_ms += Ms(Clock::now() - t0).count()
                             - (g_stats.disk_write_ms - before);
    }
    for (auto& r : runs) { r.data.clear(); r.data.shrink_to_fit(); }
}

// =========================================================
// Page writer + packing  (identical to the CUDA build)
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
        std::vector<char> zero(PAGE_BYTES, 0);
        out.write(zero.data(), PAGE_BYTES);
        bytes_written += PAGE_BYTES;
        next_page = 1;
    }
    uint64_t append(const Entry* e, uint32_t count, uint32_t is_leaf, double& ms) {
        if (used + PAGE_BYTES > buf.size()) flush(ms);
        pack_page(buf.data() + used, e, count, is_leaf);
        used += PAGE_BYTES;
        return next_page++;
    }
    double write_raw(const char* p, size_t n) {
        auto t0 = Clock::now();
        out.write(p, n); bytes_written += n;
        return Ms(Clock::now() - t0).count();
    }
    void flush(double& ms) {
        if (!used) return;
        auto t0 = Clock::now();
        out.write(buf.data(), used); bytes_written += used;
        ms += Ms(Clock::now() - t0).count(); used = 0;
    }
    void close(double& ms) { flush(ms); out.close(); }
};

static void pack_nodes(const Entry* src, size_t n, size_t cap, uint32_t is_leaf,
                       PageWriter& pw, std::vector<Entry>& parent_out, double& write_ms) {
    size_t groups = ceil_div(n, cap);
    for (size_t g = 0; g < groups; g++) {
        size_t s = g * cap, c = std::min(cap, n - s);
        Rect m = mbr_of(src + s, c);
        uint64_t page = pw.append(src + s, (uint32_t)c, is_leaf, write_ms);
        parent_out.push_back(Entry{m, page});
    }
}

static std::vector<Entry> str_pass(std::vector<Entry>& level, size_t cap,
                                   uint32_t is_leaf, PageWriter& pw, double& write_ms) {
    std::vector<Entry> parent;
    size_t n = level.size();
    if (n == 0) return parent;
    if (n <= cap) {
        auto t0 = Clock::now();
        pack_nodes(level.data(), n, cap, is_leaf, pw, parent, write_ms);
        g_stats.pack_ms += Ms(Clock::now() - t0).count();
        return parent;
    }
    sort_entries(level.data(), n, true);
    size_t slice = compute_slice_elems(n, cap, SORT_CHUNK_ELEMS);
    parent.reserve(ceil_div(n, cap));
    for (size_t off = 0; off < n; off += slice) {
        size_t cnt = std::min(slice, n - off);
        sort_entries(level.data() + off, cnt, false);
        auto t0 = Clock::now();
        pack_nodes(level.data() + off, cnt, cap, is_leaf, pw, parent, write_ms);
        g_stats.pack_ms += Ms(Clock::now() - t0).count();
    }
    return parent;
}

// =========================================================
// Builder
// =========================================================

static void build(const std::string& input, const std::string& output,
                  float fill_leaf, float fill_internal) {
    size_t file_bytes  = std::filesystem::file_size(input);
    size_t total_elems = file_bytes / sizeof(Entry);
    if (total_elems == 0) { std::cerr << "empty input\n"; std::exit(1); }

    size_t leaf_cap = apply_fill(MAX_ENTRIES_PER_PAGE, fill_leaf);
    size_t int_cap  = apply_fill(MAX_ENTRIES_PER_PAGE, fill_internal);
    size_t exp_nodes = precompute_num_nodes(total_elems, leaf_cap, int_cap);
    uint32_t exp_hgt = precompute_height(total_elems, leaf_cap, int_cap);
    size_t num_leaves = ceil_div(total_elems, leaf_cap);

    std::cout << std::fixed << std::setprecision(2)
              << "Rectangles:          " << total_elems
              << "  (" << file_bytes / (1024.0*1024.0) << " MB)\n"
              << "Threads:             " << g_threads << "\n"
              << "Effective capacity:  " << leaf_cap << " leaf, " << int_cap << " internal\n"
              << "Expected nodes:      " << exp_nodes << ", height " << exp_hgt << "\n"
              << "SORT_CHUNK_ELEMS:    " << SORT_CHUNK_ELEMS << "\n";

    std::filesystem::create_directories("tmp");
    auto wall0 = Clock::now();

    g_stats = {};
    auto tA = Clock::now();
    size_t ram_budget = USABLE_RAM_BYTES;
    std::vector<CachedRun> runs;
    generate_runs(input, total_elems, runs, ram_budget);
    TimingStats sA = g_stats; sA.wall_ms = Ms(Clock::now() - tA).count();
    sA.print("Phase A - CPU sort by centroid-X into runs");
    size_t cached = 0; for (auto& r : runs) if (!r.data.empty()) cached++;
    std::cout << "  Runs: " << runs.size() << " total, " << cached
              << " cached in RAM, " << (runs.size() - cached) << " spilled\n";

    g_stats = {};
    auto tB = Clock::now();
    const std::string sorted_x = "tmp/rect_sorted_x_cpu.bin";
    EntryVec sorted_mem; bool in_memory = false;
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
    TimingStats sB = g_stats; sB.wall_ms = Ms(Clock::now() - tB).count();
    sB.print("Phase B - k-way merge of X-sorted runs");
    std::cout << "  Merged data: " << (in_memory ? "in RAM" : "on disk") << "\n";

    g_stats = {};
    auto tC = Clock::now();
    PageWriter pw; pw.open(output);

    size_t page_bytes_per_elem = ceil_div(PAGE_BYTES, leaf_cap) + 1;
    size_t per_elem = 2 * (sizeof(Entry) + page_bytes_per_elem);
    size_t host_max = PHASE_LEAF_HOST_BYTES / per_elem;
    size_t slice = compute_slice_elems(total_elems, leaf_cap,
                                       std::min(SORT_CHUNK_ELEMS, host_max));
    size_t slice_pages = ceil_div(slice, leaf_cap) + 1;

    auto tsC = Clock::now();
    EntryVec s0, s1; s0.resize(slice); s1.resize(slice);
    Entry* sbuf[2] = { s0.data(), s1.data() };
    ByteVec pbuf0, pbuf1; pbuf0.resize(slice_pages * PAGE_BYTES); pbuf1.resize(slice_pages * PAGE_BYTES);
    char* pbuf[2] = { pbuf0.data(), pbuf1.data() };
    g_stats.setup_ms += Ms(Clock::now() - tsC).count();

    std::vector<Entry> level; level.reserve(num_leaves);
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
    g_stats.disk_read_ms += r0; g_stats.bytes_read += n0 * sizeof(Entry);

    int cur = 0, other = 1; size_t cur_n = n0; bool src_done = (n0 == 0);
    char* pend_pages = nullptr; size_t pend_bytes = 0;

    while (cur_n > 0) {
        char* wp = pend_pages; size_t wb = pend_bytes;
        bool do_read = !src_done; Entry* rdst = sbuf[other];

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

        sort_entries(sbuf[cur], cur_n, false);

        auto tp = Clock::now();
        size_t nleaves = ceil_div(cur_n, leaf_cap);
        for (size_t g = 0; g < nleaves; g++) {
            size_t s = g * leaf_cap, c = std::min(leaf_cap, cur_n - s);
            Rect m = mbr_of(sbuf[cur] + s, c);
            pack_page(pbuf[cur] + g * PAGE_BYTES, sbuf[cur] + s, (uint32_t)c, 1);
            level.push_back(Entry{m, pw.next_page++});
        }
        g_stats.pack_ms += Ms(Clock::now() - tp).count();

        pend_pages = pbuf[cur]; pend_bytes = nleaves * PAGE_BYTES;

        size_t next_n = 0;
        if (do_io) {
            auto [nr, wms, rms] = io.get();
            g_stats.disk_write_ms += wms; g_stats.bytes_written += wb;
            g_stats.disk_read_ms  += rms; g_stats.bytes_read += nr * sizeof(Entry);
            next_n = nr;
            if (nr == 0 && do_read) src_done = true;
        }
        std::swap(cur, other);
        cur_n = next_n;
    }
    if (pend_pages) {
        g_stats.disk_write_ms += pw.write_raw(pend_pages, pend_bytes);
        g_stats.bytes_written += pend_bytes;
    }
    if (in_x.is_open()) in_x.close();
    sorted_mem.clear(); sorted_mem.shrink_to_fit();
    if (!in_memory) std::filesystem::remove(sorted_x);

    TimingStats sC = g_stats; sC.wall_ms = Ms(Clock::now() - tC).count();
    sC.print("Phase C - CPU sort by centroid-Y, pack leaf pages");
    std::cout << "  Leaves packed: " << level.size() << "\n";

    g_stats = {};
    auto tD = Clock::now();
    double write_ms = 0;
    uint32_t height = 1;
    uint64_t root_page = level.empty() ? 0 : level[0].child;
    Rect root_mbr = level.empty() ? mbr_empty() : level[0].mbr;
    while (level.size() > 1) {
        std::vector<Entry> parent = str_pass(level, int_cap, 0, pw, write_ms);
        height++;
        std::cout << "  Level " << height << ": " << level.size()
                  << " entries -> " << parent.size() << " nodes\n";
        level.swap(parent);
    }
    if (!level.empty()) { root_page = level[0].child; root_mbr = level[0].mbr; }
    pw.close(write_ms);
    g_stats.disk_write_ms += write_ms;
    g_stats.bytes_written += pw.bytes_written;
    TimingStats sD = g_stats; sD.wall_ms = Ms(Clock::now() - tD).count();
    sD.print("Phase D - build internal levels");

    uint64_t num_nodes = pw.next_page - 1;
    {
        FileHeader h{};
        h.magic = RTREE_MAGIC; h.page_bytes = (uint32_t)PAGE_BYTES;
        h.height = height; h.max_entries_per_page = (uint32_t)MAX_ENTRIES_PER_PAGE;
        h.leaf_cap = (uint32_t)leaf_cap; h.internal_cap = (uint32_t)int_cap;
        h.fill_leaf = fill_leaf; h.fill_internal = fill_internal;
        h.num_rects = total_elems; h.num_nodes = num_nodes;
        h.root_page = root_page; h.root_mbr = root_mbr;
        std::fstream f(output, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(0);
        f.write(reinterpret_cast<const char*>(&h), sizeof(h));
        f.close();
    }

    auto wall1 = Clock::now();
    TimingStats total; total += sA; total += sB; total += sC; total += sD;
    total.wall_ms = Ms(wall1 - wall0).count();
    total.print("GRAND TOTAL");

    std::cout << "\nR-Tree written: " << output << "\n"
              << "  Nodes:   " << num_nodes << " pages   expected " << exp_nodes << "\n"
              << "  Height:  " << height << "   expected " << exp_hgt << "\n"
              << "\nTotal wall-clock: " << Ms(wall1 - wall0).count() / 1000.0 << " s\n";
    if (num_nodes != exp_nodes || height != exp_hgt)
        std::cerr << "WARNING: structure does not match precomputation\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input.bin> <output.bin> [options]\n\n"
                  << "  --threads N         1 = single-threaded, 0 = hardware_concurrency\n"
                  << "  --fill-leaf F       leaf fill factor      (default 1.0)\n"
                  << "  --fill-internal F   internal fill factor  (default 1.0)\n"
                  << "  --fill F            set both\n";
        return 1;
    }
    std::string input = argv[1], output = argv[2];
    float fl = DEFAULT_FILL_LEAF, fi = DEFAULT_FILL_INTERNAL;
    int threads = 1;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--threads")       threads = std::stoi(next());
        else if (a == "--fill-leaf")     fl = std::stof(next());
        else if (a == "--fill-internal") fi = std::stof(next());
        else if (a == "--fill")          fl = fi = std::stof(next());
        else { std::cerr << "unknown option: " << a << "\n"; return 1; }
    }
    if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
    g_threads = threads < 1 ? 1 : threads;
    g_merge_threads = g_threads;
    if (!std::filesystem::exists(input)) { std::cerr << "input not found\n"; return 1; }
    build(input, output, fl, fi);
    return 0;
}
