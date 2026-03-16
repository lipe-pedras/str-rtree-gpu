// rtree_query.cpp
// Point query and range query for STR R-Trees produced by
// external_str_cuda / external_str_cpu.
//
// Uses mmap for zero-copy access to the R-tree file.
//
// Usage:
//   ./rtree_query <rtree.bin> point  <x> <y>
//   ./rtree_query <rtree.bin> range  <min_x> <min_y> <max_x> <max_y> [--count]
//
// Compile: g++ -O3 -std=c++17 rtree_query.cpp -o rtree_query

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "constants.h"

// =========================================================
// Data structures (must match external_str output format)
// =========================================================

struct Point {
    float x, y;
};
static_assert(sizeof(Point) == 8);

struct MBR {
    float min_x, min_y, max_x, max_y;
};

struct RTreeNode {
    MBR      mbr;
    uint64_t first_child;
    uint32_t num_children;
    uint32_t is_leaf;
};
static_assert(sizeof(RTreeNode) == 32);

struct RTreeHeader {
    uint32_t magic;
    uint32_t leaf_capacity;
    uint32_t height;
    uint32_t internal_capacity;
    uint64_t num_points;
    uint64_t num_nodes;
};
static_assert(sizeof(RTreeHeader) == 32);

// =========================================================
// Memory-mapped R-tree handle
// =========================================================

struct RTree {
    const RTreeHeader* header;
    const RTreeNode*   nodes;
    const Point*       points;

    void*  mmap_base = nullptr;
    size_t mmap_size = 0;

    bool open(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return false; }

        struct stat st;
        fstat(fd, &st);
        mmap_size = st.st_size;

        mmap_base = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (mmap_base == MAP_FAILED) { perror("mmap"); return false; }

        // Advise sequential access for initial header read
        madvise(mmap_base, mmap_size, MADV_RANDOM);

        auto* base = reinterpret_cast<const char*>(mmap_base);
        header = reinterpret_cast<const RTreeHeader*>(base);

        if (header->magic != 0x52545245) {
            std::cerr << "Error: bad magic (not an RTRE file)\n";
            return false;
        }

        nodes  = reinterpret_cast<const RTreeNode*>(base + sizeof(RTreeHeader));
        points = reinterpret_cast<const Point*>(
                     base + sizeof(RTreeHeader)
                          + header->num_nodes * sizeof(RTreeNode));
        return true;
    }

    void close() {
        if (mmap_base && mmap_base != MAP_FAILED)
            munmap(mmap_base, mmap_size);
        mmap_base = nullptr;
    }

    size_t root_index() const { return header->num_nodes - 1; }

    void print_info() const {
        const auto& root = nodes[root_index()];
        std::cout << std::fixed << std::setprecision(4)
                  << "R-Tree: " << header->num_points << " points, "
                  << header->num_nodes << " nodes, "
                  << header->height << " levels\n"
                  << "  Leaf capacity:     " << header->leaf_capacity << "\n"
                  << "  Internal capacity: " << header->internal_capacity << "\n"
                  << "  Root MBR: [" << root.mbr.min_x << ", "
                  << root.mbr.min_y << "] × ["
                  << root.mbr.max_x << ", " << root.mbr.max_y << "]\n";
    }
};

// =========================================================
// Geometry helpers
// =========================================================

inline bool mbr_contains_point(const MBR& m, float px, float py) {
    return px >= m.min_x && px <= m.max_x &&
           py >= m.min_y && py <= m.max_y;
}

inline bool mbr_intersects(const MBR& a, const MBR& b) {
    return a.min_x <= b.max_x && a.max_x >= b.min_x &&
           a.min_y <= b.max_y && a.max_y >= b.min_y;
}

inline bool mbr_contains_mbr(const MBR& outer, const MBR& inner) {
    return inner.min_x >= outer.min_x && inner.max_x <= outer.max_x &&
           inner.min_y >= outer.min_y && inner.max_y <= outer.max_y;
}

// =========================================================
// Point query
//
// Finds all points at exactly (qx, qy).  Returns count.
// =========================================================

struct PointQueryStats {
    size_t nodes_visited = 0;
    size_t points_checked = 0;
    size_t results = 0;
};

void point_query_recurse(const RTree& rt, size_t node_idx,
                         float qx, float qy,
                         std::vector<Point>& results,
                         PointQueryStats& stats) {
    const RTreeNode& nd = rt.nodes[node_idx];
    stats.nodes_visited++;

    if (!mbr_contains_point(nd.mbr, qx, qy))
        return;

    if (nd.is_leaf) {
        // Scan points in this leaf
        for (uint32_t i = 0; i < nd.num_children; i++) {
            const Point& p = rt.points[nd.first_child + i];
            stats.points_checked++;
            if (p.x == qx && p.y == qy) {
                results.push_back(p);
                stats.results++;
            }
        }
    } else {
        // Recurse into children whose MBR contains the query point
        for (uint32_t i = 0; i < nd.num_children; i++) {
            point_query_recurse(rt, nd.first_child + i, qx, qy,
                                results, stats);
        }
    }
}

size_t point_query(const RTree& rt, float qx, float qy,
                   std::vector<Point>& results,
                   PointQueryStats& stats) {
    results.clear();
    stats = {};
    point_query_recurse(rt, rt.root_index(), qx, qy, results, stats);
    return results.size();
}

// =========================================================
// Range query
//
// Finds all points within the rectangle [min_x,max_x]×[min_y,max_y].
// If count_only is true, only counts (avoids storing large results).
// =========================================================

struct RangeQueryStats {
    size_t nodes_visited = 0;
    size_t leaves_fully_contained = 0;
    size_t points_checked = 0;
    size_t results = 0;
};

void range_query_recurse(const RTree& rt, size_t node_idx,
                         const MBR& query,
                         std::vector<Point>* results,
                         RangeQueryStats& stats) {
    const RTreeNode& nd = rt.nodes[node_idx];
    stats.nodes_visited++;

    if (!mbr_intersects(nd.mbr, query))
        return;

    if (nd.is_leaf) {
        // Optimization: if query fully contains this leaf's MBR,
        // all points match — no per-point check needed.
        if (mbr_contains_mbr(query, nd.mbr)) {
            stats.leaves_fully_contained++;
            stats.results += nd.num_children;
            if (results) {
                for (uint32_t i = 0; i < nd.num_children; i++)
                    results->push_back(rt.points[nd.first_child + i]);
            }
        } else {
            // Partial overlap — check each point
            for (uint32_t i = 0; i < nd.num_children; i++) {
                const Point& p = rt.points[nd.first_child + i];
                stats.points_checked++;
                if (p.x >= query.min_x && p.x <= query.max_x &&
                    p.y >= query.min_y && p.y <= query.max_y) {
                    stats.results++;
                    if (results) results->push_back(p);
                }
            }
        }
    } else {
        // Recurse into children whose MBR intersects the query
        for (uint32_t i = 0; i < nd.num_children; i++) {
            range_query_recurse(rt, nd.first_child + i, query,
                                results, stats);
        }
    }
}

size_t range_query(const RTree& rt, const MBR& query,
                   std::vector<Point>* results,
                   RangeQueryStats& stats) {
    if (results) results->clear();
    stats = {};
    range_query_recurse(rt, rt.root_index(), query, results, stats);
    return stats.results;
}

// =========================================================
// Main
// =========================================================

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage:\n"
                  << "  " << argv[0] << " <rtree.bin> point  <x> <y>\n"
                  << "  " << argv[0] << " <rtree.bin> range  "
                  << "<min_x> <min_y> <max_x> <max_y> [--count]\n";
        return 1;
    }

    RTree rt;
    if (!rt.open(argv[1])) return 1;
    rt.print_info();
    std::cout << "\n";

    std::string mode = argv[2];

    if (mode == "point") {
        if (argc < 5) {
            std::cerr << "point query requires <x> <y>\n";
            return 1;
        }
        float qx = std::stof(argv[3]);
        float qy = std::stof(argv[4]);

        std::cout << "Point query: (" << qx << ", " << qy << ")\n";

        std::vector<Point> results;
        PointQueryStats stats;

        auto t0 = Clock::now();
        point_query(rt, qx, qy, results, stats);
        auto t1 = Clock::now();

        std::cout << std::fixed << std::setprecision(4)
                  << "  Found:          " << stats.results << " point(s)\n"
                  << "  Nodes visited:  " << stats.nodes_visited << "\n"
                  << "  Points checked: " << stats.points_checked << "\n"
                  << "  Time:           " << Ms(t1 - t0).count() << " ms\n";

        // Print up to 20 results
        size_t show = std::min(results.size(), (size_t)20);
        for (size_t i = 0; i < show; i++)
            std::cout << "    (" << results[i].x << ", "
                      << results[i].y << ")\n";
        if (results.size() > show)
            std::cout << "    ... and " << (results.size() - show)
                      << " more\n";

    } else if (mode == "range") {
        if (argc < 7) {
            std::cerr << "range query requires "
                      << "<min_x> <min_y> <max_x> <max_y>\n";
            return 1;
        }
        MBR query;
        query.min_x = std::stof(argv[3]);
        query.min_y = std::stof(argv[4]);
        query.max_x = std::stof(argv[5]);
        query.max_y = std::stof(argv[6]);

        bool count_only = false;
        for (int i = 7; i < argc; i++) {
            if (std::string(argv[i]) == "--count")
                count_only = true;
        }

        std::cout << "Range query: [" << query.min_x << ", "
                  << query.min_y << "] × ["
                  << query.max_x << ", " << query.max_y << "]"
                  << (count_only ? "  (count only)" : "") << "\n";

        std::vector<Point> results;
        RangeQueryStats stats;

        auto t0 = Clock::now();
        range_query(rt, query, count_only ? nullptr : &results, stats);
        auto t1 = Clock::now();

        std::cout << std::fixed << std::setprecision(4)
                  << "  Found:                  " << stats.results << " point(s)\n"
                  << "  Nodes visited:          " << stats.nodes_visited << "\n"
                  << "  Leaves fully contained: " << stats.leaves_fully_contained << "\n"
                  << "  Points checked:         " << stats.points_checked << "\n"
                  << "  Time:                   " << Ms(t1 - t0).count() << " ms\n";

        if (!count_only) {
            size_t show = std::min(results.size(), (size_t)20);
            for (size_t i = 0; i < show; i++)
                std::cout << "    (" << results[i].x << ", "
                          << results[i].y << ")\n";
            if (results.size() > show)
                std::cout << "    ... and " << (results.size() - show)
                          << " more\n";
        }

    } else {
        std::cerr << "Unknown mode: " << mode
                  << "  (use 'point' or 'range')\n";
        return 1;
    }

    rt.close();
    return 0;
}
