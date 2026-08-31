# STR animation

A Manim animation of Sort-Tile-Recursive bulk loading, for the IX Simpósio de IC
presentation. It walks from 60 scattered rectangles to a finished, packed R-Tree.

```bash
make render                # 1080p60 -> animation/media/videos/str_animation/1080p60/
make render MANIM_Q=l      # fast 480p draft while iterating
make clean-anim            # remove the rendered video and generated dataset
```

Requires the Manim virtualenv at `~/.venvs/manim` (Manim CE 0.21, cairo, ffmpeg,
and a LaTeX install for the `MathTex` captions).

## It animates real output, not a mock-up

`apply_fill` in [`../src/rect_rtree_format.h`](../src/rect_rtree_format.h) floors node
capacity at 2 entries, so `--fill-leaf 0.03` drops it from 170 to **5**. Sixty
rectangles at capacity 5 therefore produce a tree small enough to draw legibly:

```
level 0:  60 entries, 4 slices of 15  ->  12 leaves     (pages 1-12)
level 1:  12 entries, 2 slices of 10  ->   3 nodes      (pages 13-15)
level 2:   3 entries                  ->   1 root       (page 16)
height 3, 100% occupancy
```

`make render` builds that tree with the **real** `bin/gen_rects` and `bin/str_rtree`,
then runs `str_reference.py --validate`, which rebuilds it in Python and compares
against the binary **page by page, entry by entry**. Rendering only proceeds if they
match. An animation that drifts from the implementation it documents is worse than no
animation, and this is the check that prevents it.

## Files

| File | Role |
|---|---|
| `rtree_reader.py` | Parses the tree binary (`FileHeader` / `PageHeader` / `Entry`) |
| `str_reference.py` | The STR pass in Python, exposing the intermediate states the file does not store, plus the validation above |
| `str_animation.py` | The Manim scene, `STRAlgorithm` |

The tree file records only the final packing, so the X-sorted order, the slice
boundaries and the Y-order inside each slice have to be recomputed — that is what
`str_reference.py` is for, and why it is validated rather than trusted.

## Visual grammar

**The rectangles never move.** STR sorts the *array*, not the geometry, so ordering is
shown as colour: a left-to-right gradient after the X sort, then one colour per slice,
then a lighter shade per group after the Y sort. Sliding rectangles around the plane
would be a much prettier animation of a different algorithm.

Levels are distinguished by colour family so the hierarchy reads: cool colours for
leaves, orange for internal nodes, white for the root.

The tree diagram on the right is drawn from the **real page numbers**. Level 1 re-sorts
the leaves before grouping them, so a parent's children are *not* contiguous page ids —
the crossing edges in the diagram are that fact, not a layout bug.

## The eight beats

1. **The input** — 60 rectangles, capacity 5
2. **Sort by centroid X** — key is `x_min + x_max` (the division is monotone, so the
   implementation skips it; the animation uses the same key)
3. **Cut into vertical slices** — `⌈√12⌉ = 4` slices of 15
4. **Sort each slice by centroid Y** — inside a slice only; this is what forms the tiles
5. **Pack groups into leaves** — 5 consecutive entries per leaf, 12 leaves
6. **Repeat with the leaves as input** — the same pass, one level up. This is the beat
   the animation exists for.
7. **Until one node is left** — 3 nodes fit one page, so that node is the root
8. **The packed R-Tree** — height 3, 100% occupancy

## Follow-ups not built

Two further animations were considered and deliberately left out: *why re-tiling at
every level matters* (long thin MBRs versus square ones, and the query cost of each),
and *the out-of-core pipeline*. The first would reuse `str_reference.py` directly.
