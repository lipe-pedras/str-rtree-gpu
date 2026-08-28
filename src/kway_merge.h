// kway_merge.h
// K-way merge strategies for Phase B, shared by the GPU and CPU loaders so the
// control group differs ONLY in where sorting happens.
//
// Chosen on measurements from bench/bench_kway_merge.cu (see context.md):
//
//   IN-RAM regime (all runs cached) — pure CPU, no I/O at all, so the
//   structure is the entire cost:
//       binary heap      ~31 M entries/s   (what this project used before)
//       loser tree       ~90 M entries/s   3x, single-threaded, drop-in
//       partition+loser ~305 M entries/s   ~10x, and FLAT in k
//   `partition` splits the OUTPUT into T disjoint key ranges and merges each
//   independently, so it parallelises perfectly and hits the memory-bandwidth
//   roof (~14.6 GB/s of traffic) rather than a comparison roof.
//
//   DISK regime (runs spilled) — measured against a cold-cache, fsync'ed I/O
//   floor, the merge costs 4-6x that floor, so it is NOT I/O-bound as one
//   might assume: reads are synchronous, so cost is I/O + CPU rather than
//   max(I/O, CPU).  The loser tree wins there too (1.2-1.4x) and needs no
//   random access into the runs, which a partitioned merge would require.
//
// A GPU merge was measured and REJECTED: with input and output both in RAM it
// must cross PCIe twice, and 2 x N over PCIe (13.3 GB/s) is slower than doing
// the whole merge in RAM (~14.6 GB/s effective).  Merging is only log2(k)
// comparisons per element — far too little work per byte to amortise the
// transfer, unlike sorting.  See context.md.

#ifndef KWAY_MERGE_H
#define KWAY_MERGE_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "rect_rtree_format.h"

// =========================================================
// Keys
// =========================================================

inline uint32_t merge_key_u(const Entry& e, bool by_x) {
    float f = by_x ? centroid_key_x(e.mbr) : centroid_key_y(e.mbr);
    uint32_t u; std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // total-order preserving
}
inline float merge_key_f(const Entry& e, bool by_x) {
    return by_x ? centroid_key_x(e.mbr) : centroid_key_y(e.mbr);
}

struct RunSpan { const Entry* p; size_t n; };

// =========================================================
// Loser tree (tournament tree)
//
// ls[0] holds the winner; ls[1..k-1] hold the LOSER of each internal match.
// After emitting the winner and advancing its run, replaying that single leaf
// to the root costs exactly ceil(log2 k) comparisons — with no container
// resize, no branch on heap size, and a fixed access pattern.  That is why it
// beats a binary heap ~3x despite doing the same asymptotic work.
// =========================================================

class LoserTree {
public:
    void init(const std::vector<RunSpan>& runs, bool by_x) {
        runs_ = &runs; by_x_ = by_x;
        k_ = (int)runs.size();
        ls_.assign(k_, k_);                       // k_ = sentinel, loses to nothing
        key_.assign(k_ + 1, 0.0f);
        pos_.assign(k_, 0);
        key_[k_] = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < k_; i++)
            key_[i] = runs[i].n ? merge_key_f(runs[i].p[0], by_x)
                                : std::numeric_limits<float>::infinity();
        for (int i = k_ - 1; i >= 0; i--) adjust(i);
    }

    size_t drain(Entry* out) {
        size_t o = 0;
        const float INF = std::numeric_limits<float>::infinity();
        for (;;) {
            int w = ls_[0];
            if (w == k_ || key_[w] == INF) break;
            out[o++] = (*runs_)[w].p[pos_[w]];
            key_[w] = (++pos_[w] < (*runs_)[w].n)
                    ? merge_key_f((*runs_)[w].p[pos_[w]], by_x_) : INF;
            adjust(w);
        }
        return o;
    }

private:
    void adjust(int s) {
        for (int t = (s + k_) / 2; t > 0; t /= 2)
            if (key_[s] > key_[ls_[t]]) std::swap(s, ls_[t]);
        ls_[0] = s;
    }
    const std::vector<RunSpan>* runs_ = nullptr;
    bool by_x_ = true;
    int k_ = 0;
    std::vector<int>    ls_;
    std::vector<float>  key_;
    std::vector<size_t> pos_;
};

// =========================================================
// In-RAM merge, partitioned across threads.
//
// The output is split into T contiguous key ranges.  Boundaries are found by
// binary search on the 32-bit ordered key: for a candidate K, count elements
// with key < K across all runs in O(k log n).  Each thread then merges its own
// disjoint slice with a loser tree, writing to a known output offset — so
// there is no locking, no atomics, and no merging of partial results.
//
// Falls back to a single loser tree for one thread or tiny inputs.
// =========================================================

inline void kway_merge_ram(const std::vector<RunSpan>& runs, Entry* out,
                           size_t total, bool by_x, int threads) {
    const size_t k = runs.size();
    if (k == 0 || total == 0) return;

    int T = threads < 1 ? 1 : threads;
    if (T == 1 || total < (1u << 16) || k == 1) {
        LoserTree lt; lt.init(runs, by_x); lt.drain(out);
        return;
    }

    auto lower_pos = [&](const RunSpan& r, uint32_t K) {
        size_t lo = 0, hi = r.n;
        while (lo < hi) {
            size_t m = lo + (hi - lo) / 2;
            if (merge_key_u(r.p[m], by_x) < K) lo = m + 1; else hi = m;
        }
        return lo;
    };
    auto count_less = [&](uint32_t K) {
        size_t c = 0;
        for (const auto& r : runs) c += lower_pos(r, K);
        return c;
    };

    std::vector<std::vector<size_t>> split(T + 1, std::vector<size_t>(k, 0));
    for (size_t i = 0; i < k; i++) split[T][i] = runs[i].n;

    for (int t = 1; t < T; t++) {
        size_t target = total * (size_t)t / (size_t)T;
        uint32_t lo = 0, hi = 0xFFFFFFFFu;
        while (lo < hi) {                       // smallest K with count_less(K) >= target
            uint32_t mid = lo + (hi - lo) / 2;
            if (count_less(mid) < target) lo = mid + 1; else hi = mid;
        }
        for (size_t i = 0; i < k; i++) split[t][i] = lower_pos(runs[i], lo);
    }

    std::vector<size_t> base(T + 1, 0);
    for (int t = 0; t <= T; t++) {
        size_t s = 0;
        for (size_t i = 0; i < k; i++) s += split[t][i];
        base[t] = s;
    }

    std::vector<std::thread> pool;
    pool.reserve(T);
    for (int t = 0; t < T; t++) {
        pool.emplace_back([&, t] {
            std::vector<RunSpan> sub(k);
            for (size_t i = 0; i < k; i++)
                sub[i] = { runs[i].p + split[t][i], split[t + 1][i] - split[t][i] };
            LoserTree lt; lt.init(sub, by_x); lt.drain(out + base[t]);
        });
    }
    for (auto& th : pool) th.join();
}

// =========================================================
// Streaming loser tree, for runs that are NOT random-access.
//
// `Cursor` must provide:
//     bool         eof()     const
//     const Entry& current() const
//     bool         advance()          // false when the run is exhausted
//
// Used for the disk-backed merge, where a partitioned merge is not available
// (it would need binary search inside each spilled file).  Measured 1.2-1.4x
// faster than a binary heap on cold-cache spilled runs, for identical I/O.
// =========================================================

template <class Cursor, class Emit>
void loser_merge_stream(std::vector<Cursor>& cur, bool by_x, Emit emit) {
    const int k = (int)cur.size();
    if (k == 0) return;
    const float INF = std::numeric_limits<float>::infinity();

    std::vector<int>   ls(k, k);
    std::vector<float> key(k + 1, 0.0f);
    key[k] = -INF;
    for (int i = 0; i < k; i++)
        key[i] = cur[i].eof() ? INF : merge_key_f(cur[i].current(), by_x);

    auto adjust = [&](int s) {
        for (int t = (s + k) / 2; t > 0; t /= 2)
            if (key[s] > key[ls[t]]) std::swap(s, ls[t]);
        ls[0] = s;
    };
    for (int i = k - 1; i >= 0; i--) adjust(i);

    for (;;) {
        int w = ls[0];
        if (w == k || key[w] == INF) break;
        emit(cur[w].current());
        key[w] = cur[w].advance() ? merge_key_f(cur[w].current(), by_x) : INF;
        adjust(w);
    }
}

#endif // KWAY_MERGE_H
