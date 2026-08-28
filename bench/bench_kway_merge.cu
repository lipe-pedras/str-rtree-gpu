// bench_kway_merge.cu
// Comparative study of k-way merge strategies for Phase B of the STR loader.
//
// WHY THIS EXISTS
//   Once the sort was moved to the GPU and made a radix sort, the single
//   biggest cost left in a 60 M-rectangle build was the k-way merge: ~2.4 s of
//   a ~5.7 s run, single-threaded, ~35 ns per entry of pointer-chasing through
//   a binary heap.  This benchmark asks whether that is fixable, and by how
//   much, before any of it is wired into the loader.
//
// THE REGIME DISTINCTION THAT DRIVES THE WHOLE STUDY
//   The merge reads every run once and writes the result once.  That is
//   already I/O-optimal, so:
//
//     * When the runs are SPILLED TO DISK, the merge cannot beat the disk.
//       Its floor is (read N + write N) bytes at device bandwidth, and the
//       comparison work hides underneath that.  A faster merge algorithm buys
//       nothing.
//     * When the runs are CACHED IN RAM, there is no I/O at all and the merge
//       is pure CPU.  Here the algorithm is the entire cost, and it is worth
//       optimizing.
//
//   So the benchmark measures BOTH, and reports the disk floor explicitly, so
//   the reader can see where effort pays and where it is wasted.
//
// STRATEGIES COMPARED (in-RAM regime)
//   heap        std::push_heap / pop_heap over k items          (current impl)
//   loser       tournament / loser tree, exactly ceil(log2 k)
//               comparisons per element, no branch on size
//   cascade     log2(k) rounds of pairwise std::merge, threaded
//   partition   exact multi-way split into T disjoint key ranges,
//               each merged independently by a loser tree, threaded
//   gpu_merge   log2(k) rounds of pairwise thrust::merge on device
//   gpu_sort    ignore the runs entirely, one radix sort_by_key on device
//               -- the "why merge at all?" upper bound
//
// Build: nvcc -O3 -std=c++17 -arch=sm_86 -I../src -Xcompiler -pthread \
//              bench_kway_merge.cu -o bench_kway_merge
// Run:   ./bench_kway_merge [--n N_MILLIONS...] [--k K...] [--threads T] [--disk]

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include <limits>

#include "rect_rtree_format.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// Total-order-preserving float -> uint32, so partitioning can binary-search on
// an integer key and terminate exactly.
__host__ __device__ inline uint32_t order_float(float f) {
    uint32_t u; memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline uint32_t keyu(const Entry& e) { return order_float(centroid_key_x(e.mbr)); }
static inline float    keyf(const Entry& e) { return centroid_key_x(e.mbr); }

struct RunSpan { const Entry* p; size_t n; };

// =========================================================
// 1. Binary heap  (what the loader does today)
// =========================================================
static void merge_heap(const std::vector<RunSpan>& runs, Entry* out) {
    struct Item { float key; size_t run, pos; };
    auto cmp = [](const Item& a, const Item& b) { return a.key > b.key; };

    std::vector<Item> heap;
    heap.reserve(runs.size());
    for (size_t i = 0; i < runs.size(); i++)
        if (runs[i].n) heap.push_back({keyf(runs[i].p[0]), i, 0});
    std::make_heap(heap.begin(), heap.end(), cmp);

    size_t o = 0;
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        Item t = heap.back(); heap.pop_back();
        out[o++] = runs[t.run].p[t.pos];
        if (++t.pos < runs[t.run].n) {
            t.key = keyf(runs[t.run].p[t.pos]);
            heap.push_back(t);
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
}

// =========================================================
// 2. Loser tree (tournament tree)
//
// The classic external-sort structure.  `ls[0]` holds the winner; `ls[1..k-1]`
// hold the LOSER of each internal match.  After emitting the winner and
// advancing its run, replaying that one leaf to the root costs exactly
// ceil(log2 k) comparisons -- no container resizing, no branch on heap size,
// and a fixed, predictable memory access pattern.
// =========================================================
struct LoserTree {
    int k;
    std::vector<int>      ls;    // ls[0] = winner index
    std::vector<float>    key;   // key[i] = current key of run i; key[k] = -inf
    std::vector<size_t>   pos;
    const std::vector<RunSpan>* runs;

    void init(const std::vector<RunSpan>& r) {
        runs = &r;
        k = (int)r.size();
        ls.assign(k, k);                       // k = sentinel that loses to nothing
        key.assign(k + 1, 0.0f);
        pos.assign(k, 0);
        key[k] = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < k; i++)
            key[i] = r[i].n ? keyf(r[i].p[0]) : std::numeric_limits<float>::infinity();
        for (int i = k - 1; i >= 0; i--) adjust(i);
    }
    void adjust(int s) {
        for (int t = (s + k) / 2; t > 0; t /= 2)
            if (key[s] > key[ls[t]]) std::swap(s, ls[t]);
        ls[0] = s;
    }
    // Emit everything into `out`; returns count written.
    size_t run_all(Entry* out) {
        size_t o = 0;
        for (;;) {
            int w = ls[0];
            if (w == k || pos[w] >= (*runs)[w].n) break;
            out[o++] = (*runs)[w].p[pos[w]];
            if (++pos[w] < (*runs)[w].n) key[w] = keyf((*runs)[w].p[pos[w]]);
            else                         key[w] = std::numeric_limits<float>::infinity();
            adjust(w);
            if (key[ls[0]] == std::numeric_limits<float>::infinity()) break;
        }
        return o;
    }
};

static void merge_loser(const std::vector<RunSpan>& runs, Entry* out) {
    LoserTree t; t.init(runs); t.run_all(out);
}

// =========================================================
// 3. Cascaded pairwise std::merge, multi-threaded
//
// log2(k) rounds; within a round the pair merges are independent.  Uses the
// heavily optimized std::merge, but moves O(N log k) bytes instead of O(N).
// =========================================================
static void merge_cascade(const std::vector<RunSpan>& runs, Entry* out,
                          Entry* scratch, size_t total, int threads) {
    auto less = [](const Entry& a, const Entry& b) { return keyf(a) < keyf(b); };

    // Materialize the runs contiguously in `scratch`, recording boundaries.
    std::vector<size_t> bnd; bnd.push_back(0);
    size_t off = 0;
    for (auto& r : runs) { memcpy(scratch + off, r.p, r.n * sizeof(Entry)); off += r.n; bnd.push_back(off); }

    Entry* src = scratch;
    Entry* dst = out;
    size_t nseg = runs.size();

    while (nseg > 1) {
        std::vector<size_t> nb; nb.push_back(0);
        std::vector<std::thread> pool;
        size_t outpos = 0;
        std::vector<std::array<size_t,4>> jobs;   // lo, mid, hi, outpos
        for (size_t i = 0; i < nseg; i += 2) {
            size_t lo = bnd[i];
            size_t mid = bnd[std::min(i + 1, nseg)];
            size_t hi  = bnd[std::min(i + 2, nseg)];
            jobs.push_back({lo, mid, hi, outpos});
            outpos += hi - lo;
            nb.push_back(outpos);
        }
        int active = std::min<int>(threads, (int)jobs.size());
        std::vector<std::thread> th;
        for (int t = 0; t < active; t++)
            th.emplace_back([&, t] {
                for (size_t j = t; j < jobs.size(); j += active) {
                    auto& J = jobs[j];
                    if (J[1] >= J[2]) std::copy(src + J[0], src + J[2], dst + J[3]);
                    else std::merge(src + J[0], src + J[1], src + J[1], src + J[2],
                                    dst + J[3], less);
                }
            });
        for (auto& x : th) x.join();
        bnd.swap(nb);
        nseg = jobs.size();
        std::swap(src, dst);
    }
    if (src != out) memcpy(out, src, total * sizeof(Entry));
}

// =========================================================
// 4. Exact range-partitioned parallel merge
//
// Split the OUTPUT into T contiguous key ranges whose boundaries are found by
// binary search on the 32-bit ordered key: for a candidate K, count elements
// with key < K across all runs in O(k log n).  Each thread then merges its own
// disjoint slice with a loser tree, writing to a known offset.  No locks, no
// synchronization, and the output is produced in place.
// =========================================================
static void merge_partition(const std::vector<RunSpan>& runs, Entry* out,
                            size_t total, int threads) {
    size_t k = runs.size();
    int T = std::max(1, threads);

    auto lower_pos = [&](const RunSpan& r, uint32_t K) {
        size_t lo = 0, hi = r.n;
        while (lo < hi) { size_t m = (lo + hi) / 2; if (keyu(r.p[m]) < K) lo = m + 1; else hi = m; }
        return lo;
    };
    auto count_less = [&](uint32_t K) {
        size_t c = 0; for (auto& r : runs) c += lower_pos(r, K); return c;
    };

    // splits[t] = per-run start offsets for partition t
    std::vector<std::vector<size_t>> splits(T + 1, std::vector<size_t>(k, 0));
    for (size_t i = 0; i < k; i++) splits[T][i] = runs[i].n;

    for (int t = 1; t < T; t++) {
        size_t target = total * t / T;
        uint32_t lo = 0, hi = 0xFFFFFFFFu;
        while (lo < hi) {                       // smallest K with count_less(K) >= target
            uint32_t mid = lo + (hi - lo) / 2;
            if (count_less(mid) < target) lo = mid + 1; else hi = mid;
        }
        for (size_t i = 0; i < k; i++) splits[t][i] = lower_pos(runs[i], lo);
    }

    std::vector<size_t> outbase(T + 1, 0);
    for (int t = 0; t <= T; t++) {
        size_t s = 0; for (size_t i = 0; i < k; i++) s += splits[t][i];
        outbase[t] = s;
    }

    std::vector<std::thread> th;
    for (int t = 0; t < T; t++)
        th.emplace_back([&, t] {
            std::vector<RunSpan> sub(k);
            for (size_t i = 0; i < k; i++)
                sub[i] = { runs[i].p + splits[t][i], splits[t + 1][i] - splits[t][i] };
            LoserTree lt; lt.init(sub); lt.run_all(out + outbase[t]);
        });
    for (auto& x : th) x.join();
}

// =========================================================
// 5/6. GPU strategies
// =========================================================
__global__ void k_make_keys(const Entry* s, uint32_t* kk, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    kk[i] = order_float(s[i].mbr.min_x + s[i].mbr.max_x);
}

struct EntryLessX {
    __host__ __device__ bool operator()(const Entry& a, const Entry& b) const {
        return (a.mbr.min_x + a.mbr.max_x) < (b.mbr.min_x + b.mbr.max_x);
    }
};

// log2(k) rounds of pairwise thrust::merge on the device.
static bool merge_gpu_pairwise(const std::vector<RunSpan>& runs, Entry* out,
                               size_t total, double& ms) {
    size_t need = 2 * total * sizeof(Entry);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (need > freeb * 9 / 10) return false;

    Entry *dA = nullptr, *dB = nullptr;
    if (cudaMalloc(&dA, total * sizeof(Entry)) != cudaSuccess) return false;
    if (cudaMalloc(&dB, total * sizeof(Entry)) != cudaSuccess) { cudaFree(dA); return false; }

    auto t0 = Clock::now();
    std::vector<size_t> bnd; bnd.push_back(0);
    size_t off = 0;
    for (auto& r : runs) {
        cudaMemcpy(dA + off, r.p, r.n * sizeof(Entry), cudaMemcpyHostToDevice);
        off += r.n; bnd.push_back(off);
    }
    Entry* src = dA; Entry* dst = dB;
    size_t nseg = runs.size();
    while (nseg > 1) {
        std::vector<size_t> nb; nb.push_back(0);
        size_t outpos = 0;
        for (size_t i = 0; i < nseg; i += 2) {
            size_t lo = bnd[i], mid = bnd[std::min(i + 1, nseg)], hi = bnd[std::min(i + 2, nseg)];
            if (mid >= hi)
                cudaMemcpy(dst + outpos, src + lo, (hi - lo) * sizeof(Entry), cudaMemcpyDeviceToDevice);
            else
                thrust::merge(thrust::device,
                              thrust::device_pointer_cast(src + lo),
                              thrust::device_pointer_cast(src + mid),
                              thrust::device_pointer_cast(src + mid),
                              thrust::device_pointer_cast(src + hi),
                              thrust::device_pointer_cast(dst + outpos), EntryLessX());
            outpos += hi - lo; nb.push_back(outpos);
        }
        cudaDeviceSynchronize();
        bnd.swap(nb); nseg = (nseg + 1) / 2; std::swap(src, dst);
    }
    cudaMemcpy(out, src, total * sizeof(Entry), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ms = Ms(Clock::now() - t0).count();
    cudaFree(dA); cudaFree(dB);
    return true;
}

// Ignore the runs' sortedness entirely: upload, radix sort, download.
static bool merge_gpu_sort(const std::vector<RunSpan>& runs, Entry* out,
                           size_t total, double& ms) {
    size_t need = total * (sizeof(Entry) + sizeof(uint32_t)) * 2;
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (need > freeb * 9 / 10) return false;

    Entry* d = nullptr; uint32_t* dk = nullptr;
    if (cudaMalloc(&d, total * sizeof(Entry)) != cudaSuccess) return false;
    if (cudaMalloc(&dk, total * sizeof(uint32_t)) != cudaSuccess) { cudaFree(d); return false; }

    auto t0 = Clock::now();
    size_t off = 0;
    for (auto& r : runs) { cudaMemcpy(d + off, r.p, r.n * sizeof(Entry), cudaMemcpyHostToDevice); off += r.n; }
    k_make_keys<<<(unsigned)((total + 255) / 256), 256>>>(d, dk, total);
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(dk),
                        thrust::device_pointer_cast(dk + total),
                        thrust::device_pointer_cast(d));
    cudaMemcpy(out, d, total * sizeof(Entry), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ms = Ms(Clock::now() - t0).count();
    cudaFree(d); cudaFree(dk);
    return true;
}

// =========================================================
// Verification
// =========================================================
static bool check_sorted(const Entry* a, size_t n, uint64_t expect_sum) {
    uint64_t sum = 0;
    for (size_t i = 1; i < n; i++)
        if (keyf(a[i - 1]) > keyf(a[i])) { printf("  !! not sorted at %zu\n", i); return false; }
    for (size_t i = 0; i < n; i++) sum += a[i].child;
    if (sum != expect_sum) { printf("  !! id checksum mismatch\n"); return false; }
    return true;
}

// =========================================================
// Disk regime: is a spilled merge I/O-bound or CPU-bound?
//
// This is the question that decides whether optimizing the merge is worth
// anything at scale.  The merge already reads each run once and writes the
// output once, which is I/O-optimal, so IF the disk were the binding
// constraint the algorithm would be irrelevant.
//
// Getting an honest answer requires defeating the page cache.  Every file is
// fsync'ed and then evicted with posix_fadvise(POSIX_FADV_DONTNEED) before
// each measurement -- otherwise "disk" bandwidth is really page-cache
// bandwidth and every number is meaningless.  Both a warm and a cold figure
// are reported so the difference is visible.
// =========================================================

// Force a file's dirty pages out and then drop it from the page cache.
// Without the fsync, a "write" merely dirties page cache and the cost shows up
// later, inside whatever is measured NEXT -- which made an earlier version of
// this benchmark report warm runs as SLOWER than cold ones.
static void fsync_path(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    ::fsync(fd);
    ::close(fd);
}

static void evict(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    ::fsync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

struct DiskReader {
    std::ifstream f; std::vector<Entry> b; size_t pos = 0, end = 0; bool at_eof = false;
    void open(const std::string& p, size_t n) { b.resize(n); f.open(p, std::ios::binary); fill(); }
    void fill() {
        f.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(Entry));
        end = f.gcount() / sizeof(Entry); pos = 0; if (!end) at_eof = true;
    }
    bool eof() const { return at_eof; }
    const Entry& cur() const { return b[pos]; }
    bool adv() { if (++pos >= end) { fill(); return !at_eof; } return true; }
};

static double disk_merge_heap(const std::vector<std::string>& paths, size_t rbuf,
                              const std::string& out) {
    auto t0 = Clock::now();
    size_t k = paths.size();
    std::vector<DiskReader> rd(k);
    for (size_t i = 0; i < k; i++) rd[i].open(paths[i], rbuf);
    struct It { float key; size_t run; };
    auto cmp = [](const It& a, const It& b) { return a.key > b.key; };
    std::vector<It> h;
    for (size_t i = 0; i < k; i++) if (!rd[i].eof()) h.push_back({keyf(rd[i].cur()), i});
    std::make_heap(h.begin(), h.end(), cmp);

    std::ofstream o(out, std::ios::binary);
    std::vector<Entry> ob(rbuf); size_t ou = 0;
    while (!h.empty()) {
        std::pop_heap(h.begin(), h.end(), cmp);
        It t = h.back(); h.pop_back();
        ob[ou++] = rd[t.run].cur();
        if (ou == ob.size()) { o.write(reinterpret_cast<char*>(ob.data()), ou * sizeof(Entry)); ou = 0; }
        if (rd[t.run].adv()) { h.push_back({keyf(rd[t.run].cur()), t.run});
                               std::push_heap(h.begin(), h.end(), cmp); }
    }
    if (ou) o.write(reinterpret_cast<char*>(ob.data()), ou * sizeof(Entry));
    o.flush(); o.close();
    fsync_path(out);                       // the write is not done until it is durable
    return Ms(Clock::now() - t0).count();
}

// Same I/O, same buffers — only the selection structure differs.
static double disk_merge_loser(const std::vector<std::string>& paths, size_t rbuf,
                               const std::string& out) {
    auto t0 = Clock::now();
    int k = (int)paths.size();
    std::vector<DiskReader> rd(k);
    for (int i = 0; i < k; i++) rd[i].open(paths[i], rbuf);

    std::vector<int>   ls(k, k);
    std::vector<float> key(k + 1, 0.0f);
    key[k] = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; i++)
        key[i] = rd[i].eof() ? std::numeric_limits<float>::infinity() : keyf(rd[i].cur());
    auto adjust = [&](int s) {
        for (int t = (s + k) / 2; t > 0; t /= 2)
            if (key[s] > key[ls[t]]) std::swap(s, ls[t]);
        ls[0] = s;
    };
    for (int i = k - 1; i >= 0; i--) adjust(i);

    std::ofstream o(out, std::ios::binary);
    std::vector<Entry> ob(rbuf); size_t ou = 0;
    for (;;) {
        int w = ls[0];
        if (w == k || key[w] == std::numeric_limits<float>::infinity()) break;
        ob[ou++] = rd[w].cur();
        if (ou == ob.size()) { o.write(reinterpret_cast<char*>(ob.data()), ou * sizeof(Entry)); ou = 0; }
        if (rd[w].adv()) key[w] = keyf(rd[w].cur());
        else             key[w] = std::numeric_limits<float>::infinity();
        adjust(w);
    }
    if (ou) o.write(reinterpret_cast<char*>(ob.data()), ou * sizeof(Entry));
    o.flush(); o.close();
    fsync_path(out);
    return Ms(Clock::now() - t0).count();
}

static void disk_regime(size_t total, size_t k, int /*threads*/) {
    double gb = 2.0 * total * sizeof(Entry) / 1e9;
    printf("\n### DISK REGIME: %zu M entries, %zu spilled runs "
           "(%.2f GB read + %.2f GB written = %.2f GB of traffic)\n",
           total / 1000000, k, gb / 2, gb / 2, gb);

    std::filesystem::create_directories("tmp");
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(0, 1000);

    size_t per = total / k;
    std::vector<std::string> paths;
    {
        std::vector<Entry> buf(per);
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < per; j++) { float x = u(rng), y = u(rng); buf[j] = {{x,y,x+1,y+1}, i*per+j}; }
            std::sort(buf.begin(), buf.end(), [](const Entry& a, const Entry& b){ return keyf(a) < keyf(b); });
            std::string p = "tmp/bench_run_" + std::to_string(i) + ".bin";
            std::ofstream f(p, std::ios::binary);
            f.write(reinterpret_cast<char*>(buf.data()), per * sizeof(Entry));
            f.close();
            paths.push_back(p);
        }
    }

    size_t rbuf = std::max<size_t>(4096, (256u << 20) / ((k + 1) * sizeof(Entry)));
    constexpr size_t CBUF = 4u << 20;

    // Quiesce: flush ALL pending writeback system-wide, then drop every file
    // involved from the page cache.  Both halves are necessary — evicting
    // without syncing just defers the write cost into the next measurement.
    auto evict_all = [&]{
        ::sync();
        for (auto& p : paths) evict(p);
        evict("tmp/bench_floor.bin");
        evict("tmp/bench_merged.bin");
    };

    auto copy_floor = [&]() {
        auto t0 = Clock::now();
        std::vector<Entry> io(CBUF / sizeof(Entry));
        std::ofstream o("tmp/bench_floor.bin", std::ios::binary);
        for (auto& p : paths) {
            std::ifstream f(p, std::ios::binary);
            while (f) {
                f.read(reinterpret_cast<char*>(io.data()), io.size() * sizeof(Entry));
                size_t n = f.gcount() / sizeof(Entry);
                if (!n) break;
                o.write(reinterpret_cast<char*>(io.data()), n * sizeof(Entry));
            }
        }
        o.flush(); o.close();
        fsync_path("tmp/bench_floor.bin");
        return Ms(Clock::now() - t0).count();
    };

    const int TRIALS = 3;
    auto best_cold = [&](auto fn) {
        double b = 1e18;
        for (int r = 0; r < TRIALS; r++) { evict_all(); b = std::min(b, fn()); }
        return b;
    };

    printf("  %-34s %11s %10s\n", "", "cold ms", "GB/s");
    double cf = best_cold([&]{ return copy_floor(); });
    printf("  %-34s %11.1f %10.2f\n", "I/O floor: copy, no merging", cf, gb / (cf / 1000));

    double hm = best_cold([&]{ return disk_merge_heap(paths, rbuf, "tmp/bench_merged.bin"); });
    printf("  %-34s %11.1f %10.2f\n", "binary heap merge", hm, gb / (hm / 1000));

    double lm = best_cold([&]{ return disk_merge_loser(paths, rbuf, "tmp/bench_merged.bin"); });
    printf("  %-34s %11.1f %10.2f\n", "loser tree merge", lm, gb / (lm / 1000));

    printf("  -> heap = %.2fx the I/O floor, loser = %.2fx; "
           "loser beats heap by %.2fx; CPU work above the floor: heap %.0f ms, loser %.0f ms\n",
           hm / cf, lm / cf, hm / lm, hm - cf, lm - cf);

    for (auto& p : paths) std::filesystem::remove(p);
    std::filesystem::remove("tmp/bench_floor.bin");
    std::filesystem::remove("tmp/bench_merged.bin");
}

// =========================================================
int main(int argc, char** argv) {
    std::vector<size_t> Ns = {4, 16, 64};
    std::vector<size_t> Ks = {2, 4, 8, 16, 32};
    int threads = (int)std::thread::hardware_concurrency();
    bool do_disk = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--n")   { Ns.clear(); while (i+1<argc && argv[i+1][0]!='-') Ns.push_back(strtoull(argv[++i],0,10)); }
        else if (a == "--k") { Ks.clear(); while (i+1<argc && argv[i+1][0]!='-') Ks.push_back(strtoull(argv[++i],0,10)); }
        else if (a == "--threads") threads = atoi(argv[++i]);
        else if (a == "--disk")    do_disk = true;
    }

    cudaFree(0);   // context creation outside every measured region
    printf("# k-way merge study — %d threads available, 24-byte Entry\n", threads);
    printf("# times in ms, best of 3 for CPU strategies; throughput in M entries/s\n");

    for (size_t nm : Ns) {
        size_t total = nm * 1000000ull;
        std::vector<Entry> src(total), out(total), scratch(total);

        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(0, 1000);
        uint64_t idsum = 0;
        for (size_t i = 0; i < total; i++) {
            float x = u(rng), y = u(rng);
            src[i] = Entry{{x, y, x + 1, y + 1}, i};
            idsum += i;
        }

        for (size_t k : Ks) {
            if (k > total) continue;
            // Build k sorted runs in place.
            size_t per = total / k;
            std::vector<RunSpan> runs;
            for (size_t i = 0; i < k; i++) {
                size_t lo = i * per, hi = (i == k - 1) ? total : (i + 1) * per;
                std::sort(src.begin() + lo, src.begin() + hi,
                          [](const Entry& a, const Entry& b) { return keyf(a) < keyf(b); });
                runs.push_back({ src.data() + lo, hi - lo });
            }

            printf("\n=== N = %zu M, k = %zu runs (%zu entries each, %.2f GB) ===\n",
                   nm, k, per, total * sizeof(Entry) / 1e9);
            printf("%-14s %10s %12s %10s %s\n", "strategy", "ms", "M ent/s", "vs heap", "ok");

            double base = 0;
            auto report = [&](const char* name, double ms, bool ok, bool avail = true) {
                if (!avail) { printf("%-14s %10s %12s %10s %s\n", name, "n/a", "-", "-", "(does not fit in VRAM)"); return; }
                if (base == 0) base = ms;
                printf("%-14s %10.1f %12.1f %9.2fx %s\n", name, ms,
                       total / (ms / 1000.0) / 1e6, base / ms, ok ? "ok" : "FAIL");
            };

            auto bestof = [&](auto fn) {
                double best = 1e18;
                for (int r = 0; r < 3; r++) {
                    auto t0 = Clock::now(); fn(); double m = Ms(Clock::now() - t0).count();
                    best = std::min(best, m);
                }
                return best;
            };

            double t;
            t = bestof([&]{ merge_heap(runs, out.data()); });
            report("heap", t, check_sorted(out.data(), total, idsum));

            t = bestof([&]{ merge_loser(runs, out.data()); });
            report("loser", t, check_sorted(out.data(), total, idsum));

            t = bestof([&]{ merge_cascade(runs, out.data(), scratch.data(), total, threads); });
            report("cascade", t, check_sorted(out.data(), total, idsum));

            t = bestof([&]{ merge_partition(runs, out.data(), total, threads); });
            report("partition", t, check_sorted(out.data(), total, idsum));

            // The rows above all merge into a buffer that is already allocated
            // and already page-faulted, because `out` is reused across trials.
            // A real loader allocates the output fresh, and std::vector::resize
            // VALUE-initialises it -- writing zeros over the whole buffer and
            // faulting every page in on a single thread before the merge can
            // start.  This row exposes that cost, which turned out to dwarf the
            // merge itself and to be invisible to every "best of N" benchmark.
            {
                double best = 1e18, best_alloc = 0;
                for (int r = 0; r < 3; r++) {
                    auto a0 = Clock::now();
                    std::vector<Entry> fresh; fresh.resize(total);
                    double am = Ms(Clock::now() - a0).count();
                    auto b0 = Clock::now();
                    merge_partition(runs, fresh.data(), total, threads);
                    double mm = Ms(Clock::now() - b0).count();
                    if (am + mm < best) { best = am + mm; best_alloc = am; }
                }
                printf("%-14s %10.1f %12.1f %9.2fx  (of which %.0f ms is resize/zero-fill)\n",
                       "  +fresh out", best, total / (best / 1000.0) / 1e6, base / best, best_alloc);
            }

            double gms = 0;
            bool ok_gpu = merge_gpu_pairwise(runs, out.data(), total, gms);
            report("gpu_merge", gms, ok_gpu && check_sorted(out.data(), total, idsum), ok_gpu);

            gms = 0;
            ok_gpu = merge_gpu_sort(runs, out.data(), total, gms);
            report("gpu_sort", gms, ok_gpu && check_sorted(out.data(), total, idsum), ok_gpu);
        }
    }

    if (do_disk) {
        for (size_t nm : Ns) disk_regime(nm * 1000000ull, 8, threads);
    }
    return 0;
}
