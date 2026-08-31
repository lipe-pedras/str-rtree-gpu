"""Reader for the R-Tree binary produced by bin/str_rtree.

Mirrors the layout defined in src/rect_rtree_format.h, whose static_asserts fix
every size:

    file = [ page 0      : FileHeader, zero-padded to PAGE_BYTES ]
           [ page 1..P-1 : node pages, PAGE_BYTES each           ]

    FileHeader  72 B   <IIIIIIffQQQffff
    PageHeader   8 B   <II    (count, is_leaf)
    Entry       24 B   <ffffQ (min_x, min_y, max_x, max_y, child)

A node stores its CHILDREN's MBRs, so `entry.child` is an object id in a leaf
page and a child page number in an internal page.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PAGE_BYTES = 4096
RTREE_MAGIC = 0x32525452  # "RTR2"

_HEADER_FMT = "<IIIIIIffQQQffff"
_PAGE_HDR_FMT = "<II"
_ENTRY_FMT = "<ffffQ"
ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)

assert struct.calcsize(_HEADER_FMT) == 72
assert struct.calcsize(_PAGE_HDR_FMT) == 8
assert ENTRY_SIZE == 24


@dataclass(frozen=True)
class Rect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def cx(self) -> float:
        """Centroid sort key on X.

        `min + max` rather than `(min + max) / 2`: the division is monotone, so
        it would be pure waste in a key computed once per element per sort.
        The C++ uses the same shortcut (centroid_key_x).
        """
        return self.min_x + self.max_x

    @property
    def cy(self) -> float:
        return self.min_y + self.max_y


@dataclass(frozen=True)
class Entry:
    mbr: Rect
    child: int


@dataclass(frozen=True)
class FileHeader:
    magic: int
    page_bytes: int
    height: int
    max_entries_per_page: int
    leaf_cap: int
    internal_cap: int
    fill_leaf: float
    fill_internal: float
    num_rects: int
    num_nodes: int
    root_page: int
    root_mbr: Rect


@dataclass(frozen=True)
class Page:
    page_id: int
    is_leaf: bool
    entries: list[Entry]


def read_entries(path: str) -> list[Entry]:
    """Read a raw dataset written by bin/gen_rects (a flat array of Entry)."""
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) % ENTRY_SIZE:
        raise ValueError(f"{path}: size {len(blob)} is not a multiple of {ENTRY_SIZE}")
    out = []
    for off in range(0, len(blob), ENTRY_SIZE):
        a, b, c, d, child = struct.unpack_from(_ENTRY_FMT, blob, off)
        out.append(Entry(Rect(a, b, c, d), child))
    return out


def read_tree(path: str) -> tuple[FileHeader, dict[int, Page]]:
    """Parse a tree file into its header and every node page, keyed by page id."""
    with open(path, "rb") as f:
        blob = f.read()

    fields = struct.unpack_from(_HEADER_FMT, blob, 0)
    hdr = FileHeader(
        magic=fields[0],
        page_bytes=fields[1],
        height=fields[2],
        max_entries_per_page=fields[3],
        leaf_cap=fields[4],
        internal_cap=fields[5],
        fill_leaf=fields[6],
        fill_internal=fields[7],
        num_rects=fields[8],
        num_nodes=fields[9],
        root_page=fields[10],
        root_mbr=Rect(*fields[11:15]),
    )

    if hdr.magic != RTREE_MAGIC:
        raise ValueError(f"{path}: bad magic 0x{hdr.magic:08x} (expected 0x{RTREE_MAGIC:08x})")
    if hdr.page_bytes != PAGE_BYTES:
        raise ValueError(f"{path}: page size {hdr.page_bytes} != {PAGE_BYTES}")

    pages: dict[int, Page] = {}
    for pid in range(1, hdr.num_nodes + 1):
        base = pid * PAGE_BYTES
        count, is_leaf = struct.unpack_from(_PAGE_HDR_FMT, blob, base)
        entries = []
        for i in range(count):
            off = base + 8 + i * ENTRY_SIZE
            a, b, c, d, child = struct.unpack_from(_ENTRY_FMT, blob, off)
            entries.append(Entry(Rect(a, b, c, d), child))
        pages[pid] = Page(pid, bool(is_leaf), entries)

    return hdr, pages


def leaf_pages(hdr: FileHeader, pages: dict[int, Page]) -> list[Page]:
    """Leaf pages in file order, which is the order the builder packed them."""
    return [p for pid, p in sorted(pages.items()) if p.is_leaf]


def internal_pages(hdr: FileHeader, pages: dict[int, Page]) -> list[Page]:
    return [p for pid, p in sorted(pages.items()) if not p.is_leaf]
