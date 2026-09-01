// rect_rtree_query.cpp
// Query and verification for the textbook-layout STR R-Tree over rectangles.
//
// The file is mmap'ed, so a node expansion is one page fault and the tree can
// be far larger than RAM.  Because a node stores its CHILDREN's MBRs, a single
// page fetch yields every pruning decision for that node's fan — no child is
// ever read merely to discover that it should be skipped.
//
// Usage:
//   ./rect_rtree_query <tree.bin> info
//   ./rect_rtree_query <tree.bin> point <x> <y>
//   ./rect_rtree_query <tree.bin> range <min_x> <min_y> <max_x> <max_y> [--count]
//   ./rect_rtree_query <tree.bin> verify <rects.bin>
//   ./rect_rtree_query <tree.bin> bench  <rects.bin> <selectivity%> [trials]
//
// Compile: g++ -O3 -std=c++17 rect_rtree_query.cpp -o rect_rtree_query

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "rect_rtree_format.h"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// =========================================================
// mmap'ed tree
// =========================================================

struct Tree {
    const char* base = nullptr;
    size_t      size = 0;
    FileHeader  hdr{};

    bool open(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { perror("fstat"); ::close(fd); return false; }
        size = st.st_size;
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (m == MAP_FAILED) { perror("mmap"); return false; }
        base = reinterpret_cast<const char*>(m);
        madvise(const_cast<char*>(base), size, MADV_RANDOM);

        std::memcpy(&hdr, base, sizeof(FileHeader));
        if (hdr.magic != RTREE_MAGIC) {
            std::cerr << "bad magic 0x" << std::hex << hdr.magic
                      << " (expected 0x" << RTREE_MAGIC << ")\n" << std::dec;
            return false;
        }
        if (hdr.page_bytes != PAGE_BYTES) {
            std::cerr << "page size mismatch: file " << hdr.page_bytes
                      << ", build " << PAGE_BYTES << "\n";
            return false;
        }
        return true;
    }

    void close() { if (base) munmap(const_cast<char*>(base), size); base = nullptr; }

    const char* page(uint64_t id) const { return base + page_offset(id); }

    void print_info() const {
        std::cout << std::fixed << std::setprecision(4)
                  << "R-Tree: " << hdr.num_rects << " rectangles, "
                  << hdr.num_nodes << " pages, height " << hdr.height << "\n"
                  << "  Page size:          " << hdr.page_bytes << " B\n"
                  << "  Entries/page (max): " << hdr.max_entries_per_page << "\n"
                  << "  Effective capacity: " << hdr.leaf_cap << " leaf, "
                  << hdr.internal_cap << " internal\n"
                  << "  Fill factors:       " << hdr.fill_leaf << " leaf, "
                  << hdr.fill_internal << " internal\n"
                  << "  Root page:          " << hdr.root_page << "\n"
                  << "  Root MBR:           [" << hdr.root_mbr.min_x << ", "
                  << hdr.root_mbr.min_y << "] x [" << hdr.root_mbr.max_x << ", "
                  << hdr.root_mbr.max_y << "]\n"
                  << "  File size:          " << size / (1024.0*1024.0) << " MB\n";
    }
};

// =========================================================
// Query statistics
// =========================================================

struct QueryStats {
    size_t pages_read        = 0;   // node pages fetched
    size_t entries_tested    = 0;   // MBR tests performed
    size_t entries_pruned    = 0;   // children rejected without being fetched
    size_t leaves_fully_in   = 0;   // leaves entirely inside the query
    size_t results           = 0;
};

// Range query.  The parent-stored MBRs mean a child is rejected BEFORE its page
// is touched, so `entries_pruned` counts page fetches that never happened.
static void range_recurse(const Tree& t, uint64_t page_id, const Rect& q,
                          std::vector<Entry>* out, QueryStats& st) {
    const char* p = t.page(page_id);
    const PageHeader* h = page_hdr(p);
    const Entry* e = page_entries(p);
    st.pages_read++;

    if (h->is_leaf) {
        for (uint32_t i = 0; i < h->count; i++) {
            st.entries_tested++;
            if (mbr_intersects(e[i].mbr, q)) {
                st.results++;
                if (out) out->push_back(e[i]);
            } else st.entries_pruned++;
        }
    } else {
        for (uint32_t i = 0; i < h->count; i++) {
            st.entries_tested++;
            if (mbr_intersects(e[i].mbr, q)) {
                range_recurse(t, e[i].child, q, out, st);
            } else {
                st.entries_pruned++;    // subtree skipped without any I/O
            }
        }
    }
}

static void point_recurse(const Tree& t, uint64_t page_id, float x, float y,
                          std::vector<Entry>* out, QueryStats& st) {
    const char* p = t.page(page_id);
    const PageHeader* h = page_hdr(p);
    const Entry* e = page_entries(p);
    st.pages_read++;

    for (uint32_t i = 0; i < h->count; i++) {
        st.entries_tested++;
        if (!mbr_contains_point(e[i].mbr, x, y)) { st.entries_pruned++; continue; }
        if (h->is_leaf) { st.results++; if (out) out->push_back(e[i]); }
        else point_recurse(t, e[i].child, x, y, out, st);
    }
}

// =========================================================
// Verification
// =========================================================

struct VerifyState {
    std::vector<uint64_t> seen;      // bitmap over object ids
    size_t leaf_pages = 0, internal_pages = 0;
    size_t entries_total = 0;
    size_t errors = 0;
    uint32_t leaf_cap = 0, internal_cap = 0;
    uint64_t num_rects = 0;
    uint64_t max_page = 0;

    void err(const std::string& m) {
        if (errors < 20) std::cerr << "  ERROR: " << m << "\n";
        errors++;
    }
};

// Returns the union of this subtree's entry MBRs, so the caller can check that
// the MBR its parent stored for this node is correct.
static Rect verify_recurse(const Tree& t, uint64_t page_id, uint32_t depth,
                           VerifyState& v) {
    if (page_id == 0 || page_id > v.max_page) {
        v.err("page id " + std::to_string(page_id) + " out of range");
        return mbr_empty();
    }
    const char* p = t.page(page_id);
    const PageHeader* h = page_hdr(p);
    const Entry* e = page_entries(p);

    uint32_t cap = h->is_leaf ? v.leaf_cap : v.internal_cap;
    if (h->count == 0)
        v.err("page " + std::to_string(page_id) + " has zero entries");
    if (h->count > cap)
        v.err("page " + std::to_string(page_id) + " has " + std::to_string(h->count)
              + " entries, exceeds effective capacity " + std::to_string(cap));

    v.entries_total += h->count;
    Rect acc = mbr_empty();

    if (h->is_leaf) {
        v.leaf_pages++;
        for (uint32_t i = 0; i < h->count; i++) {
            mbr_extend(acc, e[i].mbr);
            uint64_t id = e[i].child;
            if (id >= v.num_rects) {
                v.err("object id " + std::to_string(id) + " out of range");
                continue;
            }
            uint64_t w = id >> 6, b = 1ull << (id & 63);
            if (v.seen[w] & b) v.err("object id " + std::to_string(id) + " appears twice");
            v.seen[w] |= b;
        }
    } else {
        v.internal_pages++;
        for (uint32_t i = 0; i < h->count; i++) {
            Rect child = verify_recurse(t, e[i].child, depth + 1, v);
            // The MBR the parent stored must exactly cover the child subtree.
            if (!mbr_contains(e[i].mbr, child))
                v.err("page " + std::to_string(page_id) + " entry " + std::to_string(i)
                      + ": stored MBR does not contain child subtree");
            mbr_extend(acc, e[i].mbr);
        }
    }
    return acc;
}

static int cmd_verify(Tree& t, const char* rect_path) {
    std::cout << "Verifying tree structure...\n";

    VerifyState v;
    v.leaf_cap     = t.hdr.leaf_cap;
    v.internal_cap = t.hdr.internal_cap;
    v.num_rects    = t.hdr.num_rects;
    v.max_page     = t.hdr.num_nodes;
    v.seen.assign((t.hdr.num_rects + 63) / 64, 0);

    auto t0 = Clock::now();
    Rect root = verify_recurse(t, t.hdr.root_page, 1, v);
    double walk_ms = Ms(Clock::now() - t0).count();

    if (!mbr_contains(t.hdr.root_mbr, root))
        v.err("header root MBR does not contain the tree");

    size_t missing = 0;
    for (uint64_t id = 0; id < t.hdr.num_rects; id++)
        if (!(v.seen[id >> 6] & (1ull << (id & 63)))) missing++;
    if (missing) v.err(std::to_string(missing) + " object ids are missing from the tree");

    size_t pages = v.leaf_pages + v.internal_pages;
    if (pages != t.hdr.num_nodes)
        v.err("walked " + std::to_string(pages) + " pages, header says "
              + std::to_string(t.hdr.num_nodes));

    std::cout << std::fixed << std::setprecision(2)
              << "  Pages walked:   " << pages << " ("
              << v.leaf_pages << " leaf, " << v.internal_pages << " internal)\n"
              << "  Entries:        " << v.entries_total << "\n"
              << "  Objects indexed:" << (t.hdr.num_rects - missing)
              << " / " << t.hdr.num_rects << "\n"
              << "  Leaf occupancy: "
              << (v.leaf_pages ? 100.0 * (t.hdr.num_rects - missing)
                                 / (double)(v.leaf_pages * t.hdr.leaf_cap) : 0.0)
              << " % of effective capacity\n"
              << "  Walk time:      " << walk_ms << " ms\n";

    // Cross-check leaf geometry against the source file, if given.
    if (rect_path) {
        std::cout << "Cross-checking against " << rect_path << " ...\n";
        std::ifstream in(rect_path, std::ios::binary);
        if (!in) { std::cerr << "  cannot open source file\n"; v.errors++; }
        else {
            // Random spot-check: verify that a sample of source rectangles is
            // returned by a range query over its own extent.
            std::mt19937_64 rng(1234);
            size_t samples = std::min<size_t>(1000, t.hdr.num_rects);
            std::uniform_int_distribution<uint64_t> pick(0, t.hdr.num_rects - 1);
            size_t not_found = 0;
            for (size_t s = 0; s < samples; s++) {
                uint64_t id = pick(rng);
                in.seekg((std::streamoff)(id * sizeof(Entry)));
                Entry src{};
                in.read(reinterpret_cast<char*>(&src), sizeof(Entry));
                if (src.child != id) { v.err("source file id mismatch at " + std::to_string(id)); break; }

                std::vector<Entry> hits;
                QueryStats st;
                range_recurse(t, t.hdr.root_page, src.mbr, &hits, st);
                bool found = false;
                for (auto& e : hits) if (e.child == id) { found = true; break; }
                if (!found) not_found++;
            }
            if (not_found) v.err(std::to_string(not_found) + " / "
                                 + std::to_string(samples)
                                 + " sampled rectangles were not returned by their own range query");
            else std::cout << "  " << samples << " sampled rectangles all round-trip correctly\n";
        }
    }

    if (v.errors == 0) { std::cout << "\nVERIFY OK\n"; return 0; }
    std::cout << "\nVERIFY FAILED: " << v.errors << " error(s)\n";
    return 2;
}

// =========================================================
// Benchmark: tree range query vs. brute-force scan
// =========================================================

static int cmd_bench(Tree& t, const char* rect_path, double sel_pct, int trials) {
    const Rect& R = t.hdr.root_mbr;
    double W = R.max_x - R.min_x, H = R.max_y - R.min_y;
    // side of a square covering sel_pct of the root area
    double side = std::sqrt(sel_pct / 100.0) ;
    double qw = W * side, qh = H * side;

    std::mt19937_64 rng(9876);
    double lox = R.min_x, hix = std::max<double>(R.min_x, (double)R.max_x - qw);
    double loy = R.min_y, hiy = std::max<double>(R.min_y, (double)R.max_y - qh);
    std::uniform_real_distribution<double> ux(lox, hix);
    std::uniform_real_distribution<double> uy(loy, hiy);

    std::cout << std::fixed << std::setprecision(4)
              << "Range-query benchmark: " << sel_pct << "% of root area, "
              << trials << " trials\n"
              << "  query box: " << qw << " x " << qh << "\n";

    double total_ms = 0;
    size_t total_pages = 0, total_tested = 0, total_pruned = 0, total_hits = 0;

    for (int i = 0; i < trials; i++) {
        Rect q;
        q.min_x = (float)ux(rng); q.min_y = (float)uy(rng);
        q.max_x = (float)(q.min_x + qw); q.max_y = (float)(q.min_y + qh);

        QueryStats st;
        auto a = Clock::now();
        range_recurse(t, t.hdr.root_page, q, nullptr, st);
        total_ms += Ms(Clock::now() - a).count();

        total_pages  += st.pages_read;
        total_tested += st.entries_tested;
        total_pruned += st.entries_pruned;
        total_hits   += st.results;
    }

    std::cout << "  avg time:          " << total_ms / trials << " ms\n"
              << "  avg pages read:    " << (double)total_pages / trials << "\n"
              << "  avg MBR tests:     " << (double)total_tested / trials << "\n"
              << "  avg pruned:        " << (double)total_pruned / trials
              << "  (" << (total_tested ? 100.0 * total_pruned / total_tested : 0.0)
              << "% of tests)\n"
              << "  avg hits:          " << (double)total_hits / trials << "\n"
              << "  bytes read:        "
              << (double)total_pages * PAGE_BYTES / trials / 1024.0 << " KB/query\n"
              << "  tests per page:    "
              << (total_pages ? (double)total_tested / total_pages : 0.0)
              << "  (pruning yield per page fetch)\n";

    // Brute-force cross-check on the first query, if the source file is given.
    if (rect_path) {
        std::ifstream in(rect_path, std::ios::binary);
        if (in) {
            Rect q;
            std::mt19937_64 rng2(9876);
            std::uniform_real_distribution<double> vx(lox, hix);
            std::uniform_real_distribution<double> vy(loy, hiy);
            q.min_x = (float)vx(rng2); q.min_y = (float)vy(rng2);
            q.max_x = (float)(q.min_x + qw); q.max_y = (float)(q.min_y + qh);

            QueryStats st;
            auto a = Clock::now();
            range_recurse(t, t.hdr.root_page, q, nullptr, st);
            double tree_ms = Ms(Clock::now() - a).count();

            size_t brute = 0;
            std::vector<Entry> buf(1 << 16);
            auto b = Clock::now();
            while (in) {
                in.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(Entry));
                size_t n = in.gcount() / sizeof(Entry);
                for (size_t i = 0; i < n; i++)
                    if (mbr_intersects(buf[i].mbr, q)) brute++;
                if (n == 0) break;
            }
            double brute_ms = Ms(Clock::now() - b).count();

            std::cout << "\n  Cross-check on one query:\n"
                      << "    tree:        " << st.results << " hits in "
                      << tree_ms << " ms\n"
                      << "    brute force: " << brute << " hits in "
                      << brute_ms << " ms\n"
                      << "    "
                      << (st.results == brute ? "MATCH" : "*** MISMATCH ***")
                      << "   speedup " << (tree_ms > 0 ? brute_ms / tree_ms : 0.0)
                      << "x\n";
            if (st.results != brute) return 2;
        }
    }
    return 0;
}

// =========================================================
// main
// =========================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout <<
        "Usage:\n"
        "  " << argv[0] << " <tree.bin> info\n"
        "  " << argv[0] << " <tree.bin> point <x> <y>\n"
        "  " << argv[0] << " <tree.bin> range <min_x> <min_y> <max_x> <max_y> [--count]\n"
        "  " << argv[0] << " <tree.bin> verify [rects.bin]\n"
        "  " << argv[0] << " <tree.bin> bench  <rects.bin|-> <selectivity%> [trials]\n";
        return 1;
    }

    Tree t;
    if (!t.open(argv[1])) return 1;
    std::string mode = argv[2];

    if (mode == "info") { t.print_info(); t.close(); return 0; }

    t.print_info();
    std::cout << "\n";

    int rc = 0;

    if (mode == "point") {
        if (argc < 5) { std::cerr << "point needs <x> <y>\n"; return 1; }
        float x = std::stof(argv[3]), y = std::stof(argv[4]);
        std::vector<Entry> out;
        QueryStats st;
        auto a = Clock::now();
        point_recurse(t, t.hdr.root_page, x, y, &out, st);
        double ms = Ms(Clock::now() - a).count();
        std::cout << std::fixed << std::setprecision(4)
                  << "Point query (" << x << ", " << y << ")\n"
                  << "  Found:         " << st.results << "\n"
                  << "  Pages read:    " << st.pages_read << "\n"
                  << "  MBR tests:     " << st.entries_tested << "\n"
                  << "  Pruned:        " << st.entries_pruned << "\n"
                  << "  Time:          " << ms << " ms\n";
        for (size_t i = 0; i < std::min<size_t>(out.size(), 20); i++)
            std::cout << "    id " << out[i].child << "  ["
                      << out[i].mbr.min_x << ", " << out[i].mbr.min_y << "] x ["
                      << out[i].mbr.max_x << ", " << out[i].mbr.max_y << "]\n";

    } else if (mode == "range") {
        if (argc < 7) { std::cerr << "range needs <min_x> <min_y> <max_x> <max_y>\n"; return 1; }
        Rect q{std::stof(argv[3]), std::stof(argv[4]),
               std::stof(argv[5]), std::stof(argv[6])};
        bool count_only = false;
        for (int i = 7; i < argc; i++) if (std::string(argv[i]) == "--count") count_only = true;

        std::vector<Entry> out;
        QueryStats st;
        auto a = Clock::now();
        range_recurse(t, t.hdr.root_page, q, count_only ? nullptr : &out, st);
        double ms = Ms(Clock::now() - a).count();

        std::cout << std::fixed << std::setprecision(4)
                  << "Range query [" << q.min_x << ", " << q.min_y << "] x ["
                  << q.max_x << ", " << q.max_y << "]"
                  << (count_only ? "  (count only)" : "") << "\n"
                  << "  Found:         " << st.results << "\n"
                  << "  Pages read:    " << st.pages_read << "\n"
                  << "  MBR tests:     " << st.entries_tested << "\n"
                  << "  Pruned:        " << st.entries_pruned << "\n"
                  << "  Bytes read:    " << st.pages_read * PAGE_BYTES / 1024.0 << " KB\n"
                  << "  Time:          " << ms << " ms\n";
        for (size_t i = 0; i < std::min<size_t>(out.size(), 20); i++)
            std::cout << "    id " << out[i].child << "\n";
        if (out.size() > 20) std::cout << "    ... and " << out.size() - 20 << " more\n";

    } else if (mode == "verify") {
        rc = cmd_verify(t, argc >= 4 ? argv[3] : nullptr);

    } else if (mode == "bench") {
        if (argc < 5) { std::cerr << "bench needs <rects.bin|-> <selectivity%>\n"; return 1; }
        const char* rp = std::string(argv[3]) == "-" ? nullptr : argv[3];
        double sel = std::stod(argv[4]);
        int trials = argc >= 6 ? std::stoi(argv[5]) : 20;
        rc = cmd_bench(t, rp, sel, trials);

    } else {
        std::cerr << "unknown mode: " << mode << "\n";
        rc = 1;
    }

    t.close();
    return rc;
}
