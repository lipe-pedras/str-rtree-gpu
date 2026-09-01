"""Manim scene: Sort-Tile-Recursive (STR) bulk loading, step by step.

Driven by a real tree built by bin/str_rtree over 60 rectangles with a node
capacity of 5 (see `make render`). str_reference.py is checked against that
tree before rendering, so what is drawn here is what the code actually does --
including the parent/child edges of the tree diagram, which come from the real
page numbers rather than being assumed contiguous.

The point the animation exists to make: the same sort-tile-pack pass repeats at
every level until one node remains. Beats 2-4 therefore replay, compressed, as
beat 5.

A note on the visual grammar: the rectangles never move. STR sorts the *array*,
not the geometry, so ordering is encoded as colour -- a left-to-right gradient
after the X sort, and a shade per group after the Y sort. Sliding the
rectangles around would misrepresent the algorithm.
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    BLUE_D, DOWN, GREEN_D, GREY_B, GREY_D, LEFT, RIGHT, TEAL_C, UP, WHITE,
    YELLOW_D, AnimationGroup, Create, DashedLine, FadeIn, FadeOut, Line,
    MathTex, ORANGE, Rectangle, Scene, Text, VGroup, Write, interpolate_color,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtree_reader import Entry, read_entries, read_tree  # noqa: E402
from str_reference import str_build  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
RECTS = DATA_DIR / "rects60.bin"
TREE = DATA_DIR / "tree60.bin"

PLOT_SIDE = 5.6
PLOT_CENTER = LEFT * 3.5 + DOWN * 0.55
PANEL_LEFT = 1.2
PANEL_WIDTH = 5.4
ROW_Y = {0: -2.75, 1: -0.75, 2: 1.30}      # leaves, internal, root

SLICE_COLORS = [BLUE_D, TEAL_C, GREEN_D, YELLOW_D]
# Internal nodes get a warm colour so the levels read as a hierarchy rather
# than as one flat pile of boxes in the same cool palette.
NODE_COLOR = ORANGE


class STRAlgorithm(Scene):
    def construct(self):
        self.camera.background_color = "#0e1117"

        objects = read_entries(str(RECTS))
        hdr, _pages = read_tree(str(TREE))
        self.hdr = hdr
        traces = str_build(objects, hdr.leaf_cap, hdr.internal_cap)
        self.leaf_trace, self.node_trace, self.root_trace = traces

        # Scale from the real data bounds, not an assumed [0,100] box: a
        # rectangle placed at x=100 with width 6 reaches 106, and clipping it
        # would be a lie about where the MBR is.
        r = hdr.root_mbr
        self.dx0, self.dy0 = r.min_x, r.min_y
        span = max(r.max_x - r.min_x, r.max_y - r.min_y)
        self.scale = PLOT_SIDE / span
        self.span = span

        self.step = VGroup()
        self.detail = VGroup()
        self.tree_nodes: dict[int, Rectangle] = {}

        self.beat_intro(objects, hdr)
        self.beat_sort_x()
        self.beat_slice()
        self.beat_sort_y()
        self.beat_pack_leaves()
        self.beat_recurse()
        self.beat_root()
        self.beat_outro(hdr)

    # -- geometry ------------------------------------------------------------

    def to_scene(self, x: float, y: float):
        return (PLOT_CENTER
                + RIGHT * ((x - self.dx0) * self.scale - PLOT_SIDE / 2)
                + UP * ((y - self.dy0) * self.scale - PLOT_SIDE / 2))

    def rect_mobject(self, e: Entry, **kw) -> Rectangle:
        m = e.mbr
        w = max((m.max_x - m.min_x) * self.scale, 0.03)
        h = max((m.max_y - m.min_y) * self.scale, 0.03)
        return Rectangle(width=w, height=h, **kw).move_to(
            self.to_scene((m.min_x + m.max_x) / 2, (m.min_y + m.max_y) / 2))

    def mbr_box(self, e: Entry, pad: float = 0.03, **kw) -> Rectangle:
        m = e.mbr
        w = (m.max_x - m.min_x) * self.scale + 2 * pad
        h = (m.max_y - m.min_y) * self.scale + 2 * pad
        return Rectangle(width=w, height=h, **kw).move_to(
            self.to_scene((m.min_x + m.max_x) / 2, (m.min_y + m.max_y) / 2))

    def divider(self, left_cx: float, right_cx: float) -> DashedLine:
        """A slice boundary, drawn midway between the two centroids it separates."""
        xm = (left_cx / 2 + right_cx / 2) / 2
        top = self.to_scene(xm, self.dy0 + self.span)
        bot = self.to_scene(xm, self.dy0)
        return DashedLine(bot, top, stroke_color=WHITE, stroke_width=2.2,
                          dash_length=0.09)

    # -- captions ------------------------------------------------------------

    def set_step(self, title: str, detail: str = "", math: str | None = None):
        """Replace the caption. Uses fade in/out rather than .become(), which
        interpolates between glyph sets and renders as overlapping garbage."""
        new_step = Text(title, font_size=30, color=WHITE).move_to(UP * 3.5)
        parts = VGroup()
        if detail:
            parts.add(Text(detail, font_size=21, color=GREY_B))
        if math:
            parts.add(MathTex(math, font_size=28, color=TEAL_C))
        if len(parts):
            parts.arrange(DOWN, buff=0.14).move_to(UP * 2.95)

        out = [FadeOut(m, shift=UP * 0.15) for m in (self.step, self.detail) if len(m)]
        if out:
            self.play(*out, run_time=0.35)
        self.step, self.detail = VGroup(new_step), parts
        anims = [FadeIn(new_step, shift=DOWN * 0.15)]
        if len(parts):
            anims.append(FadeIn(parts, shift=DOWN * 0.15))
        self.play(*anims, run_time=0.55)

    # -- beats ---------------------------------------------------------------

    def beat_intro(self, objects, hdr):
        title = Text("Sort-Tile-Recursive (STR) bulk loading", font_size=42)
        sub = Text("packing an R-Tree bottom-up, one level at a time",
                   font_size=24, color=GREY_B).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=DOWN * 0.2), run_time=0.8)
        self.wait(1.6)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.7)

        self.frame = Rectangle(width=PLOT_SIDE + 0.5, height=PLOT_SIDE + 0.5,
                               stroke_color=GREY_D, stroke_width=2).move_to(PLOT_CENTER)
        self.play(Create(self.frame), run_time=0.8)

        self.rects = VGroup(*[
            self.rect_mobject(e, stroke_color=GREY_B, stroke_width=1.6,
                              fill_color=GREY_D, fill_opacity=0.35)
            for e in objects])
        self.by_id = {e.child: m for e, m in zip(objects, self.rects)}

        self.set_step("The input",
                      f"{hdr.num_rects} rectangles  -  {hdr.leaf_cap} entries per node")
        self.play(AnimationGroup(*[FadeIn(m, scale=0.7) for m in self.rects],
                                 lag_ratio=0.025), run_time=2.2)
        self.wait(1.2)

    def beat_sort_x(self):
        self.set_step("1 - Sort by centroid X",
                      "the array is reordered - the geometry stays put",
                      r"\text{key} = x_{\min} + x_{\max}")
        order = self.leaf_trace.x_sorted
        anims = [self.by_id[e.child].animate
                 .set_stroke(interpolate_color(BLUE_D, YELLOW_D, i / (len(order) - 1)),
                             width=2.0)
                 .set_fill(interpolate_color(BLUE_D, YELLOW_D, i / (len(order) - 1)),
                           opacity=0.45)
                 for i, e in enumerate(order)]
        self.play(AnimationGroup(*anims, lag_ratio=0.03), run_time=3.2)
        self.wait(1.3)

    def beat_slice(self):
        t = self.leaf_trace
        self.set_step("2 - Cut into vertical slices",
                      f"{len(t.source)} / {t.cap} = {len(t.groups)} leaves,  then "
                      f"{t.num_slices} slices of {t.slice_width} rectangles",
                      r"\lceil \sqrt{12} \rceil = 4 \text{ slices}")

        self.slice_lines = VGroup(*[
            self.divider(t.slices[k - 1][-1].mbr.cx, t.slices[k][0].mbr.cx)
            for k in range(1, t.num_slices)])
        self.play(AnimationGroup(*[Create(l) for l in self.slice_lines],
                                 lag_ratio=0.25), run_time=1.7)

        anims = []
        for si, sl in enumerate(t.slices):
            col = SLICE_COLORS[si % len(SLICE_COLORS)]
            anims += [self.by_id[e.child].animate.set_stroke(col, width=2.0)
                      .set_fill(col, opacity=0.45) for e in sl]
        self.play(AnimationGroup(*anims, lag_ratio=0.014), run_time=2.2)
        self.wait(1.3)

    def beat_sort_y(self):
        t = self.leaf_trace
        self.set_step("3 - Sort each slice by centroid Y",
                      "inside a slice only - this is what forms the tiles",
                      r"\text{key} = y_{\min} + y_{\max}")
        per_slice = len(t.groups) // t.num_slices
        anims = []
        for si, sl in enumerate(t.y_sorted):
            col = SLICE_COLORS[si % len(SLICE_COLORS)]
            for i, e in enumerate(sl):
                c = interpolate_color(col, WHITE, 0.5 * (i // t.cap) / max(per_slice - 1, 1))
                anims.append(self.by_id[e.child].animate
                             .set_stroke(c, width=2.0).set_fill(c, opacity=0.5))
        self.play(AnimationGroup(*anims, lag_ratio=0.022), run_time=2.8)
        self.wait(1.3)

    def beat_pack_leaves(self):
        t = self.leaf_trace
        self.set_step("4 - Pack groups into leaves",
                      f"{t.cap} consecutive entries per leaf  ->  {len(t.groups)} leaves")
        per_slice = len(t.groups) // t.num_slices

        self.leaf_boxes = {}
        anims = []
        for gi, parent in enumerate(t.parents):
            col = SLICE_COLORS[(gi // per_slice) % len(SLICE_COLORS)]
            box = self.mbr_box(parent, stroke_color=col, stroke_width=3.0, fill_opacity=0)
            self.leaf_boxes[parent.child] = box
            anims.append(Create(box))
        self.play(AnimationGroup(*anims, lag_ratio=0.13), run_time=3.0)

        self.draw_row(0, [p.child for p in t.parents],
                      [SLICE_COLORS[(i // per_slice) % len(SLICE_COLORS)]
                       for i in range(len(t.parents))])
        self.wait(1.4)

    def beat_recurse(self):
        t = self.node_trace
        self.set_step("5 - Now repeat, with the leaves as input",
                      f"same pass: sort X, slice, sort Y, pack  -  "
                      f"{t.num_slices} slices of {t.slice_width}")

        self.play(self.rects.animate.set_stroke(opacity=0.15).set_fill(opacity=0.05),
                  FadeOut(self.slice_lines), run_time=1.0)

        anims = [self.leaf_boxes[e.child].animate.set_stroke(
                    interpolate_color(BLUE_D, YELLOW_D, i / (len(t.x_sorted) - 1)),
                    width=3.0)
                 for i, e in enumerate(t.x_sorted)]
        self.play(AnimationGroup(*anims, lag_ratio=0.07), run_time=2.0)

        lines = VGroup(*[self.divider(t.slices[k - 1][-1].mbr.cx, t.slices[k][0].mbr.cx)
                         for k in range(1, t.num_slices)])
        self.play(AnimationGroup(*[Create(l) for l in lines], lag_ratio=0.25),
                  run_time=1.0)

        anims = []
        for si, sl in enumerate(t.y_sorted):
            col = SLICE_COLORS[si % len(SLICE_COLORS)]
            anims += [self.leaf_boxes[e.child].animate.set_stroke(col, width=3.0)
                      for e in sl]
        self.play(AnimationGroup(*anims, lag_ratio=0.06), run_time=1.8)

        self.node_boxes = {}
        anims = []
        for parent in t.parents:
            box = self.mbr_box(parent, pad=0.09, stroke_color=NODE_COLOR,
                               stroke_width=4.0, fill_opacity=0)
            self.node_boxes[parent.child] = box
            anims.append(Create(box))
        self.play(AnimationGroup(*anims, lag_ratio=0.22), run_time=2.0)
        # Push the finished level back so the new one is legible on top of it.
        self.play(*[b.animate.set_stroke(opacity=0.35, width=2.0)
                    for b in self.leaf_boxes.values()], run_time=0.7)

        self.draw_row(1, [p.child for p in t.parents],
                      [NODE_COLOR] * len(t.parents))
        self.draw_edges(1, t)
        self.play(FadeOut(lines), run_time=0.4)
        self.wait(1.4)

    def beat_root(self):
        t = self.root_trace
        self.set_step("6 - Until one node is left",
                      f"{len(t.source)} nodes fit in a single page  ->  that is the root")
        self.play(*[b.animate.set_stroke(opacity=0.55, width=3.0)
                    for b in self.node_boxes.values()],
                  self.frame.animate.set_stroke(opacity=0.25), run_time=0.7)
        root = self.mbr_box(t.parents[0], pad=0.15, stroke_color=WHITE,
                            stroke_width=5.0, fill_opacity=0)
        self.play(Create(root), run_time=1.6)
        self.root_box = root
        self.draw_row(2, [t.parents[0].child], [WHITE])
        self.draw_edges(2, t)
        self.wait(1.6)

    def beat_outro(self, hdr):
        self.set_step("The packed R-Tree",
                      f"height {hdr.height}  -  100% occupancy  -  "
                      f"every node is one {hdr.page_bytes // 1024} KB page")
        self.play(self.rects.animate.set_stroke(opacity=0.45).set_fill(opacity=0.2),
                  *[b.animate.set_stroke(opacity=0.7) for b in self.leaf_boxes.values()],
                  *[b.animate.set_stroke(opacity=0.9) for b in self.node_boxes.values()],
                  run_time=1.2)
        self.wait(3.5)

    # -- tree panel ----------------------------------------------------------

    def draw_row(self, depth: int, page_ids: list[int], colors: list):
        """One row of the tree diagram, keyed by the real page numbers."""
        y = ROW_Y[depth]
        n = len(page_ids)
        node_w = min(0.34, PANEL_WIDTH / (n * 1.55))
        xs = ([PANEL_LEFT + PANEL_WIDTH / 2] if n == 1
              else [PANEL_LEFT + i * PANEL_WIDTH / (n - 1) for i in range(n)])

        row = VGroup()
        for pid, x, col in zip(page_ids, xs, colors):
            node = Rectangle(width=node_w, height=0.26, stroke_color=col,
                             stroke_width=2.4, fill_color=col, fill_opacity=0.3)
            node.move_to(RIGHT * x + UP * y)
            self.tree_nodes[pid] = node
            row.add(node)
        self.play(AnimationGroup(*[FadeIn(n_, scale=0.6) for n_ in row], lag_ratio=0.06),
                  run_time=1.0)

    def draw_edges(self, depth: int, trace):
        """Edges from real page numbers: trace.groups[i] holds the child entries
        of the node whose page is trace.parents[i].child. The level-1 pass
        re-sorts the leaves, so a parent's children are NOT contiguous page ids
        and must not be assumed to be."""
        edges = VGroup()
        for grp, parent in zip(trace.groups, trace.parents):
            pnode = self.tree_nodes.get(parent.child)
            if pnode is None:
                continue
            for child in grp:
                cnode = self.tree_nodes.get(child.child)
                if cnode is not None:
                    edges.add(Line(pnode.get_bottom(), cnode.get_top(),
                                   stroke_color=GREY_D, stroke_width=1.5))
        if len(edges):
            self.play(AnimationGroup(*[Create(e) for e in edges], lag_ratio=0.03),
                      run_time=1.0)
