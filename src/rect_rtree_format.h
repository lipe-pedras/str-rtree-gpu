// rect_rtree_format.h
// On-disk format for the textbook-layout STR R-Tree over rectangles.
// Shared by the builder, the query tool and the generator so that all three
// agree on the layout by construction (every size is static_assert-ed).
//
//   file = [ page 0        : FileHeader, zero-padded to PAGE_BYTES ]
//          [ page 1..P-1   : Node pages, PAGE_BYTES each           ]
//
// TEXTBOOK NODE LAYOUT
//   A node is an array of entries, and an entry is (child MBR, child id).
//   A node therefore stores the MBRs of its CHILDREN, not its own.  One page
//   fetch yields every pruning decision for that node's whole subtree fan —
//   the parent IS the index over its children, which is the entire point of an
//   internal node in a disk-resident R-Tree.
//
//   The root has no parent, so its MBR is the one MBR with nowhere to live:
//   it is kept in the file header.  That is the honest 16-byte cost of this
//   layout, and it buys back the pruning-index property.
//
//   For a LEAF page,     entry.child is the ID of the indexed object.
//   For an INTERNAL page, entry.child is the page number of the child node.

#ifndef RECT_RTREE_FORMAT_H
#define RECT_RTREE_FORMAT_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "rect_constants.h"

// =========================================================
// Geometry
// =========================================================

struct Rect {
    float min_x, min_y, max_x, max_y;
};
static_assert(sizeof(Rect) == 16, "Rect must be 16 bytes");

// The single type that flows through the entire STR recursion.
//
//   level 0 : { object MBR , object id  }   <- read from the input file
//   level 1 : { leaf   MBR , leaf  page }   <- produced when leaves are packed
//   level 2 : { node   MBR , node  page }   <- and so on, up to the root
//
// Because the packed-node entry and the sort element are the SAME 24 bytes,
// each level's output array is directly the next level's sort input.  No
// extraction step, no scratch array, and — crucially — no second pass over the
// raw data to recover MBRs that were never kept.  The MBRs live in RAM between
// levels; only the packed pages ever reach the disk.
struct Entry {
    Rect     mbr;
    uint64_t child;
};
static_assert(sizeof(Entry) == 24, "Entry must be 24 bytes");

// =========================================================
// Page layout
// =========================================================

struct PageHeader {
    uint32_t count;    // entries actually stored in this page
    uint32_t is_leaf;  // 1 = leaf page, 0 = internal page
};
static_assert(sizeof(PageHeader) == 8, "PageHeader must be 8 bytes");

// Physical ceiling on entries per page.  The effective capacity is this scaled
// by the fill factor (see FileHeader::leaf_cap / internal_cap).
constexpr size_t MAX_ENTRIES_PER_PAGE =
    (PAGE_BYTES - sizeof(PageHeader)) / sizeof(Entry);   // 4088 / 24 = 170

static_assert(MAX_ENTRIES_PER_PAGE >= 2,
              "PAGE_BYTES too small to hold a usable node");

// =========================================================
// File header (lives alone in page 0)
// =========================================================

constexpr uint32_t RTREE_MAGIC = 0x32525452u;  // "RTR2"

struct FileHeader {
    uint32_t magic;
    uint32_t page_bytes;
    uint32_t height;                // 1 == root is a leaf
    uint32_t max_entries_per_page;

    uint32_t leaf_cap;              // effective entries per leaf     (after fill)
    uint32_t internal_cap;          // effective entries per internal (after fill)
    float    fill_leaf;
    float    fill_internal;

    uint64_t num_rects;
    uint64_t num_nodes;             // total pages excluding page 0
    uint64_t root_page;             // explicit — never a positional convention

    Rect     root_mbr;              // the one MBR with no parent to live in
};
static_assert(sizeof(FileHeader) <= PAGE_BYTES,
              "FileHeader must fit in page 0");

// =========================================================
// Helpers shared by builder and reader
// =========================================================

inline const PageHeader* page_hdr(const void* page) {
    return reinterpret_cast<const PageHeader*>(page);
}

inline const Entry* page_entries(const void* page) {
    return reinterpret_cast<const Entry*>(
        reinterpret_cast<const char*>(page) + sizeof(PageHeader));
}

// Serialize `count` entries into a full PAGE_BYTES page image.
inline void pack_page(char* dst, const Entry* src,
                      uint32_t count, uint32_t is_leaf) {
    std::memset(dst, 0, PAGE_BYTES);
    PageHeader h{count, is_leaf};
    std::memcpy(dst, &h, sizeof(h));
    std::memcpy(dst + sizeof(PageHeader), src, count * sizeof(Entry));
}

// Byte offset of a page in the file.  Page 0 is the header, so node pages
// start at PAGE_BYTES and every node is page-aligned by construction.
inline uint64_t page_offset(uint64_t page_id) {
    return page_id * PAGE_BYTES;
}

// =========================================================
// MBR arithmetic
// =========================================================

inline Rect mbr_empty() {
    return Rect{ 3.402823466e+38f,  3.402823466e+38f,
                -3.402823466e+38f, -3.402823466e+38f };
}

inline void mbr_extend(Rect& acc, const Rect& r) {
    if (r.min_x < acc.min_x) acc.min_x = r.min_x;
    if (r.min_y < acc.min_y) acc.min_y = r.min_y;
    if (r.max_x > acc.max_x) acc.max_x = r.max_x;
    if (r.max_y > acc.max_y) acc.max_y = r.max_y;
}

inline Rect mbr_of(const Entry* e, size_t n) {
    Rect m = mbr_empty();
    for (size_t i = 0; i < n; i++) mbr_extend(m, e[i].mbr);
    return m;
}

inline bool mbr_intersects(const Rect& a, const Rect& b) {
    return a.min_x <= b.max_x && a.max_x >= b.min_x &&
           a.min_y <= b.max_y && a.max_y >= b.min_y;
}

inline bool mbr_contains(const Rect& outer, const Rect& inner) {
    return inner.min_x >= outer.min_x && inner.max_x <= outer.max_x &&
           inner.min_y >= outer.min_y && inner.max_y <= outer.max_y;
}

inline bool mbr_contains_point(const Rect& m, float x, float y) {
    return x >= m.min_x && x <= m.max_x && y >= m.min_y && y <= m.max_y;
}

// Centroid comparison keys.  `min + max` is monotonically equivalent to
// `(min + max) / 2`, so the division is skipped — it would be pure waste in a
// key computed once per element per sort.
inline float centroid_key_x(const Rect& r) { return r.min_x + r.max_x; }
inline float centroid_key_y(const Rect& r) { return r.min_y + r.max_y; }

// =========================================================
// Capacity / shape arithmetic
// =========================================================

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// Effective entries per node after applying a fill factor.  Always >= 2, or
// the tree would never converge to a root.
inline size_t apply_fill(size_t max_cap, float fill) {
    if (fill <= 0.0f || fill > 1.0f) fill = 1.0f;
    size_t c = (size_t)(max_cap * (double)fill);
    if (c < 2) c = 2;
    if (c > max_cap) c = max_cap;
    return c;
}

// Total pages the tree will occupy — pure arithmetic, no data access needed.
inline size_t precompute_num_nodes(size_t num_rects,
                                   size_t leaf_cap,
                                   size_t internal_cap) {
    if (num_rects == 0) return 0;
    size_t level = ceil_div(num_rects, leaf_cap);
    size_t total = level;
    while (level > 1) {
        level = ceil_div(level, internal_cap);
        total += level;
    }
    return total;
}

inline uint32_t precompute_height(size_t num_rects,
                                  size_t leaf_cap,
                                  size_t internal_cap) {
    if (num_rects == 0) return 0;
    size_t level = ceil_div(num_rects, leaf_cap);
    uint32_t h = 1;
    while (level > 1) {
        level = ceil_div(level, internal_cap);
        h++;
    }
    return h;
}

// STR slice width, in entries.
//
// `⌈√(groups)⌉` slices, rounded UP to a whole multiple of `cap` so that a node
// never straddles a slice boundary, then capped at what one sort can hold —
// rounded DOWN, again to a multiple of `cap`, so the alignment survives the cap.
inline size_t compute_slice_elems(size_t level_count,
                                  size_t cap,
                                  size_t max_slice_elems) {
    size_t num_groups = ceil_div(level_count, cap);
    if (num_groups <= 1) return level_count;

    size_t num_slices  = (size_t)std::sqrt((double)num_groups);
    if (num_slices < 1) num_slices = 1;
    while (num_slices * num_slices < num_groups) num_slices++;

    size_t groups_per_slice = ceil_div(num_groups, num_slices);
    size_t slice = groups_per_slice * cap;

    size_t capped = (max_slice_elems / cap) * cap;   // keep slice/node alignment
    if (capped < cap) capped = cap;
    if (slice > capped) slice = capped;
    return slice;
}

#endif // RECT_RTREE_FORMAT_H
