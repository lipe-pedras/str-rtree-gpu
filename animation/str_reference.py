"""A Python mirror of the STR pass, used to drive the animation.

The tree file records only the FINAL packing. The animation needs the states in
between — the X-sorted order, where the slice boundaries fall, and the Y-sorted
order inside each slice — so those are recomputed here.

That creates a risk: a reimplementation can drift from the implementation it
claims to illustrate. `validate()` closes it by rebuilding the same dataset and
asserting that every page this module produces matches the page the real
bin/str_rtree wrote, entry for entry. `make render` runs that check before
rendering, so an animation can never show an algorithm the code does not run.

Mirrors compute_slice_elems() in src/rect_rtree_format.h and str_pass() in
src/str_rtree_cuda.cu.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtree_reader import Entry, Rect, leaf_pages, internal_pages, read_entries, read_tree

# Matches MAX_ENTRIES_PER_PAGE in src/rect_rtree_format.h: (4096 - 8) / 24.
MAX_ENTRIES_PER_PAGE = (4096 - 8) // 24


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def apply_fill(max_cap: int, fill: float) -> int:
    """Effective node capacity after a fill factor. Floors at 2, or the tree
    would never converge to a root."""
    if fill <= 0.0 or fill > 1.0:
        fill = 1.0
    c = int(max_cap * fill)
    return max(2, min(c, max_cap))


def compute_slice_elems(level_count: int, cap: int, max_slice_elems: int) -> int:
    """Slice width in entries.

    ceil(sqrt(groups)) slices, each rounded UP to a whole multiple of `cap` so a
    node never straddles a slice boundary, then capped at what one sort can hold
    — rounded DOWN, again to a multiple of `cap`, so the alignment survives.
    """
    num_groups = ceil_div(level_count, cap)
    if num_groups <= 1:
        return level_count

    num_slices = int(math.sqrt(num_groups))
    if num_slices < 1:
        num_slices = 1
    while num_slices * num_slices < num_groups:
        num_slices += 1

    groups_per_slice = ceil_div(num_groups, num_slices)
    width = groups_per_slice * cap

    capped = (max_slice_elems // cap) * cap
    if capped < cap:
        capped = cap
    return min(width, capped)


def mbr_of(entries: list[Entry]) -> Rect:
    return Rect(
        min(e.mbr.min_x for e in entries),
        min(e.mbr.min_y for e in entries),
        max(e.mbr.max_x for e in entries),
        max(e.mbr.max_y for e in entries),
    )


@dataclass
class LevelTrace:
    """Every intermediate state of one STR pass, for the animation to replay."""

    level: int                          # 0 = objects -> leaves
    cap: int
    is_leaf: bool
    single_node: bool                   # n <= cap: no sorting, just pack the root
    source: list[Entry] = field(default_factory=list)
    x_sorted: list[Entry] = field(default_factory=list)
    slice_width: int = 0
    slices: list[list[Entry]] = field(default_factory=list)          # after X sort
    y_sorted: list[list[Entry]] = field(default_factory=list)        # after Y sort
    groups: list[list[Entry]] = field(default_factory=list)          # one per node
    parents: list[Entry] = field(default_factory=list)               # (node MBR, page)

    @property
    def num_slices(self) -> int:
        return len(self.slices)


def str_pass(level: list[Entry], cap: int, max_slice: int,
             next_page: int, level_index: int, is_leaf: bool) -> tuple[LevelTrace, int]:
    """One STR pass: sort by X, tile, sort each tile by Y, pack groups of `cap`."""
    t = LevelTrace(level=level_index, cap=cap, is_leaf=is_leaf,
                   single_node=len(level) <= cap, source=list(level))

    if t.single_node:
        t.groups = [list(level)]
        t.parents = [Entry(mbr_of(level), next_page)]
        return t, next_page + 1

    t.x_sorted = sorted(level, key=lambda e: e.mbr.cx)
    t.slice_width = compute_slice_elems(len(t.x_sorted), cap, max_slice)
    t.slices = [t.x_sorted[i:i + t.slice_width]
                for i in range(0, len(t.x_sorted), t.slice_width)]
    t.y_sorted = [sorted(s, key=lambda e: e.mbr.cy) for s in t.slices]

    for s in t.y_sorted:
        for i in range(0, len(s), cap):
            t.groups.append(s[i:i + cap])

    for g in t.groups:
        t.parents.append(Entry(mbr_of(g), next_page))
        next_page += 1

    return t, next_page


def str_build(objects: list[Entry], leaf_cap: int, internal_cap: int,
              max_slice: int = 1 << 40) -> list[LevelTrace]:
    """Full bottom-up build, returning one trace per level.

    Matches the builder's control flow: the leaf level is always packed, then
    internal levels are packed while more than one node remains.
    """
    traces: list[LevelTrace] = []
    next_page = 1

    t, next_page = str_pass(objects, leaf_cap, max_slice, next_page, 0, True)
    traces.append(t)
    level = t.parents

    idx = 1
    while len(level) > 1:
        t, next_page = str_pass(level, internal_cap, max_slice, next_page, idx, False)
        traces.append(t)
        level = t.parents
        idx += 1

    return traces


def _same_rect(a: Rect, b: Rect) -> bool:
    return (a.min_x, a.min_y, a.max_x, a.max_y) == (b.min_x, b.min_y, b.max_x, b.max_y)


def validate(tree_path: str, data_path: str, verbose: bool = True) -> bool:
    """Rebuild with this module and compare against the real tree, page by page."""
    hdr, pages = read_tree(tree_path)
    objects = read_entries(data_path)

    if len(objects) != hdr.num_rects:
        print(f"FAIL: dataset has {len(objects)} objects, tree header says {hdr.num_rects}")
        return False

    traces = str_build(objects, hdr.leaf_cap, hdr.internal_cap)

    # Pages the builder wrote, in file order: leaves first, then each internal
    # level, then the root — which is exactly the order str_build produces.
    expected = [g for t in traces for g in t.groups]
    actual = [p for _, p in sorted(pages.items())]

    ok = True
    if len(expected) != len(actual):
        print(f"FAIL: reference packed {len(expected)} nodes, tree has {len(actual)}")
        return False

    for i, (grp, page) in enumerate(zip(expected, actual), start=1):
        if len(grp) != len(page.entries):
            print(f"FAIL: page {i} has {len(page.entries)} entries, reference has {len(grp)}")
            ok = False
            continue
        for j, (ref, got) in enumerate(zip(grp, page.entries)):
            if ref.child != got.child:
                print(f"FAIL: page {i} entry {j}: child {got.child} != reference {ref.child}")
                ok = False
            elif not _same_rect(ref.mbr, got.mbr):
                print(f"FAIL: page {i} entry {j}: MBR {got.mbr} != reference {ref.mbr}")
                ok = False

    root_ref = traces[-1].parents[0]
    if not _same_rect(root_ref.mbr, hdr.root_mbr):
        print(f"FAIL: root MBR {hdr.root_mbr} != reference {root_ref.mbr}")
        ok = False
    if len(traces) != hdr.height:
        print(f"FAIL: reference height {len(traces)} != header height {hdr.height}")
        ok = False

    if ok and verbose:
        print(f"reference matches {tree_path} exactly:")
        print(f"  {hdr.num_rects} objects, capacity {hdr.leaf_cap} leaf / "
              f"{hdr.internal_cap} internal, height {hdr.height}")
        for t in traces:
            kind = "leaves" if t.is_leaf else "nodes"
            if t.single_node:
                print(f"  level {t.level}: {len(t.source)} entries -> 1 root {kind}")
            else:
                print(f"  level {t.level}: {len(t.source)} entries, "
                      f"{t.num_slices} slices of {t.slice_width} -> {len(t.groups)} {kind}")
        print(f"  {len(actual)} pages compared, all identical")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", metavar="TREE.bin", required=True)
    ap.add_argument("--data", metavar="RECTS.bin", default=None,
                    help="dataset the tree was built from (default: sibling rects*.bin)")
    args = ap.parse_args()

    tree = Path(args.validate)
    data = Path(args.data) if args.data else tree.with_name(
        tree.name.replace("tree", "rects"))
    if not data.exists():
        print(f"dataset not found: {data}")
        return 1

    return 0 if validate(str(tree), str(data)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
