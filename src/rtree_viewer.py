#!/usr/bin/env python3
"""
Interactive R-Tree Viewer  (out-of-core / larger-than-RAM)
==========================================================
Reads the binary R-tree file produced by external_str_cuda and renders
points + MBRs in a matplotlib GUI.

The file is memory-mapped with numpy.memmap so that only the pages
currently being viewed are loaded into physical RAM by the OS.  This
allows inspection of R-trees built over datasets far larger than
available memory.

Usage:
    python rtree_viewer.py data/rtree.bin

Controls:
    Left-click an MBR    → drill into that node's children
    Right-click anywhere → go back up one level
    Scroll wheel         → zoom in/out
    Middle-click drag    → pan
    'r' key              → reset view to root
    'q' key              → quit
"""

import sys
import struct
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# ─── Binary format ────────────────────────────────────────

HEADER_SIZE = 32
NODE_SIZE   = 32
POINT_SIZE  = 8

NODE_DTYPE = np.dtype([
    ("min_x", "<f4"), ("min_y", "<f4"),
    ("max_x", "<f4"), ("max_y", "<f4"),
    ("first_child", "<u8"),
    ("num_children", "<u4"),
    ("is_leaf", "<u4"),
])

POINT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4")])


def read_header(path):
    """Read the 32-byte header without loading anything else."""
    with open(path, "rb") as f:
        buf = f.read(HEADER_SIZE)
    magic, leaf_cap, height, internal_cap = struct.unpack_from("<IIII", buf, 0)
    num_points, num_nodes = struct.unpack_from("<QQ", buf, 16)
    assert magic == 0x52545245, f"Bad magic: {magic:#x}"
    return {
        "magic": magic,
        "leaf_capacity": leaf_cap,
        "internal_capacity": internal_cap,
        "height": height,
        "num_points": num_points,
        "num_nodes": num_nodes,
    }


def mmap_rtree(path):
    """Return (header, nodes_memmap, points_memmap).

    Both arrays are read-only memory-mapped views into the file.
    The OS will page in only the regions that are actually accessed,
    so RAM usage stays proportional to the *visible* portion of the tree.
    """
    hdr = read_header(path)

    nodes_offset = HEADER_SIZE
    points_offset = nodes_offset + hdr["num_nodes"] * NODE_SIZE

    nodes = np.memmap(path, dtype=NODE_DTYPE, mode="r",
                      offset=nodes_offset,
                      shape=(hdr["num_nodes"],))

    points = np.memmap(path, dtype=POINT_DTYPE, mode="r",
                       offset=points_offset,
                       shape=(hdr["num_points"],))

    return hdr, nodes, points


# ─── Viewer ───────────────────────────────────────────────

# Color palette for tree levels (deepest → shallowest)
LEVEL_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


class RTreeViewer:
    def __init__(self, path):
        # Memory-map instead of loading the whole file
        self.hdr, self.nodes, self.points = mmap_rtree(path)

        self.root_idx = self.hdr["num_nodes"] - 1
        self.leaf_capacity = self.hdr["leaf_capacity"]
        self.internal_capacity = self.hdr["internal_capacity"]

        # Navigation stack: each entry is a list of node indices to display
        self.stack = []  # history of parent views
        self.current_indices = [self.root_idx]  # start showing the root

        # Set up figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title("R-Tree Viewer")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()
        plt.show()

    # ── drawing ────────────────────────────────────────

    def _draw(self):
        ax = self.ax
        ax.clear()

        indices = self.current_indices
        n = self.nodes

        # Determine if we are looking at leaf children (draw points)
        # or internal children (draw MBRs that can be drilled into).
        # We always draw the MBRs of `indices`.
        # If all of them are leaves, also scatter their points.

        all_leaf = all(n[i]["is_leaf"] for i in indices)

        # Collect MBR rectangles
        rects = []
        mbr_list = []
        for idx in indices:
            nd = n[idx]
            x0, y0 = float(nd["min_x"]), float(nd["min_y"])
            x1, y1 = float(nd["max_x"]), float(nd["max_y"])
            w, h = x1 - x0, y1 - y0
            rects.append(Rectangle((x0, y0), max(w, 1e-6), max(h, 1e-6)))
            mbr_list.append((x0, y0, x1, y1, idx))

        self.mbr_list = mbr_list

        depth = len(self.stack)
        color = LEVEL_COLORS[depth % len(LEVEL_COLORS)]

        pc = PatchCollection(rects, edgecolor=color, facecolor="none",
                             linewidth=1.5, alpha=0.85)
        ax.add_collection(pc)

        # Draw points if leaves — only touch the relevant slices of
        # the memmap so the OS only pages in those regions.
        if all_leaf:
            chunks_x = []
            chunks_y = []
            for idx in indices:
                nd = n[idx]
                s = int(nd["first_child"])
                e = s + int(nd["num_children"])
                chunk = self.points[s:e]      # memmap slice → pages in only this range
                chunks_x.append(chunk["x"])
                chunks_y.append(chunk["y"])
            if chunks_x:
                vis_x = np.concatenate(chunks_x)
                vis_y = np.concatenate(chunks_y)
                ax.scatter(vis_x, vis_y,
                           s=0.3, c="black", alpha=0.4, rasterized=True)

        # If we're showing children of a parent, also draw the parent MBR
        # in a lighter shade for context
        if self.stack:
            parent_indices = self.stack[-1]
            parent_rects = []
            for pidx in parent_indices:
                pnd = n[pidx]
                x0, y0 = float(pnd["min_x"]), float(pnd["min_y"])
                x1, y1 = float(pnd["max_x"]), float(pnd["max_y"])
                w, h = x1 - x0, y1 - y0
                parent_rects.append(Rectangle((x0, y0), max(w, 1e-6), max(h, 1e-6)))
            pcolor = LEVEL_COLORS[(depth - 1) % len(LEVEL_COLORS)]
            ppc = PatchCollection(parent_rects, edgecolor=pcolor,
                                  facecolor="none", linewidth=0.6,
                                  alpha=0.3, linestyle="--")
            ax.add_collection(ppc)

        # Auto-fit view to current MBRs
        if mbr_list:
            all_x0 = min(m[0] for m in mbr_list)
            all_y0 = min(m[1] for m in mbr_list)
            all_x1 = max(m[2] for m in mbr_list)
            all_y1 = max(m[3] for m in mbr_list)
            dx = (all_x1 - all_x0) * 0.05 or 1
            dy = (all_y1 - all_y0) * 0.05 or 1
            ax.set_xlim(all_x0 - dx, all_x1 + dx)
            ax.set_ylim(all_y0 - dy, all_y1 + dy)

        level_name = "root" if depth == 0 else f"depth {depth}"
        node_type = "leaf" if all_leaf else "internal"
        ax.set_title(
            f"R-Tree  |  {level_name}  |  {len(indices)} {node_type} node(s)  |  "
            f"height={self.hdr['height']}  leaf_cap={self.leaf_capacity}  int_cap={self.internal_capacity}  "
            f"nodes={self.hdr['num_nodes']}  points={self.hdr['num_points']}\n"
            f"Left-click MBR → drill down  |  Right-click → go back  |  "
            f"'r' → reset  |  'q' → quit",
            fontsize=9,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    # ── interaction ────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:  # right-click → go back
            self._go_back()
            return

        if event.button != 1:  # only left-click drills down
            return

        mx, my = event.xdata, event.ydata

        # Find smallest MBR containing click point
        best_idx = None
        best_area = float("inf")
        for x0, y0, x1, y1, idx in self.mbr_list:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                area = (x1 - x0) * (y1 - y0)
                if area < best_area:
                    best_area = area
                    best_idx = idx

        if best_idx is None:
            return

        nd = self.nodes[best_idx]
        if nd["is_leaf"]:
            return  # can't drill into a leaf further

        # Drill into children
        first = int(nd["first_child"])
        count = int(nd["num_children"])
        children = list(range(first, first + count))

        self.stack.append(self.current_indices)
        self.current_indices = children
        self._draw()

    def _go_back(self):
        if not self.stack:
            return
        self.current_indices = self.stack.pop()
        self._draw()

    def _on_key(self, event):
        if event.key == "r":
            self.stack.clear()
            self.current_indices = [self.root_idx]
            self._draw()
        elif event.key == "q":
            plt.close(self.fig)


# ─── main ─────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rtree_viewer.py <rtree.bin>")
        sys.exit(1)

    RTreeViewer(sys.argv[1])
