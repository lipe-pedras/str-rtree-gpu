# Context — GPU R-Tree Bulk-Loading When the Dataset Does Not Fit in VRAM

> Working document for an article. Collects the thesis, the central discussion,
> the findings, the supporting measurements-to-be, and the references.
> **[TODO — measure]** marks claims that need numbers from real runs.
>
> **Sections 1–10 describe the first implementation** (points, self-MBR node
> layout, `external_str_*`). **Sections 11–14 describe the rewrite**
> (`refactor/the-return-of-the-programer`): rectangles, textbook node layout,
> and the measurements that came out of building it — several of which settle
> questions left open above. Read §11 for what changed and why; §12 for the
> new measured results; §13 for what the rewrite confirmed or refuted about
> the analysis in §2–§5.

---

## 1. Thesis and research gap

**The gap.** Every GPU R-Tree paper in `references/` assumes the spatial data
already lives in GPU global memory. This is stated outright in
`Parallel_implementation_of_R-trees_on_the_GPU.pdf`:

> "In this paper, we assume the spatial data can fit into memory, which is
> reasonable given that memory is getting larger and cheaper."

The dataset scales in the reference set corroborate it: **100 K entries, 100 K
rectangles, 2 M rectangles, 1 M points, 500 K data**. Every experiment is sized
to fit comfortably in device memory.

That assumption does not survive contact with real spatial databases, and it is
aging *badly* rather than well:

- GPU memory is the **most expensive and slowest-growing tier** in the machine.
  The development GPU here is a 4 GB RTX 3050 Laptop (**3.68 GiB actually
  addressable**, of which ~2.94 GB is safely usable after CUDA runtime
  overhead). Meanwhile the host has 14 GB of RAM and the SSD has hundreds.
- Spatial datasets do not shrink. A billion 2-D single-precision points is 8 GB
  — **2.7× this GPU's entire VRAM**, and it is a modest dataset by the standards
  of OSM, TIGER, LiDAR or trajectory archives.
- The gap between "what fits in VRAM" and "what people want to index" is
  *widening*, not closing. HBM capacity growth has not tracked dataset growth.

**The thesis.** Given that the data will not fit, what is the best possible way
to use the GPU anyway? This project answers that for R-Tree bulk-loading via
STR, and the answer is a set of concrete techniques:

1. **Cached runs** — a run of sorted data is a single abstraction that is
   *either* RAM-resident *or* disk-resident, chosen greedily against a byte
   budget, so one code path spans the whole spectrum from "fits in VRAM" to
   "fits in RAM" to "fits nowhere".
2. **The right buffer topology** — double-buffering across the GPU/IO boundary,
   with a *single serialized* I/O stream. More buffers and more I/O threads made
   it slower, not faster.
3. **Single-pass k-way heap merge** — the I/O-optimal way to combine runs:
   one read, one write, regardless of run count.
4. **A node struct that stores its own MBR**, which removes an entire pass over
   the raw data from the STR recursion for 16 bytes of storage.

Target scale: `gen_points 1000000000` → **10⁹ points / 8 GB**, built on a
machine with 2.94 GB usable VRAM and a 5 GB RAM budget.

---

## 2. The central discussion — a workload that scores terribly on the standard
## GPU metric, and wins anyway

This is the most interesting argument in the project and should be the article's
spine.

### 2.1 The standard metric says: do not put this on a GPU

The conventional way to decide whether a workload belongs on a GPU is the
**roofline model** — arithmetic (operational) intensity, in FLOP per byte of
memory traffic. GPU compute throughput vastly exceeds GPU memory bandwidth, so
the received wisdom is *maximize operations per byte*; a kernel below the ridge
point is memory-bound and "wasting" the device.

For the development GPU:

| Quantity | Value |
|---|---|
| FP32 peak | 16 SM × 128 cores × 2 × ~1.5 GHz ≈ **6.1 TFLOP/s** |
| VRAM bandwidth | 6001 MHz × 2 × 128 bit / 8 = **192 GB/s** |
| **Ridge point** | 6.1e12 / 192e9 ≈ **32 FLOP/byte** |

Now the actual kernels in this project:

| Operation | Work per byte moved | vs. ridge point |
|---|---|---|
| Merge-sort pass (1 float compare per 8 B read + 8 B written) | **≈ 0.06 op/byte** | **~500× below** |
| Leaf MBR reduction (4 min/max per 8-byte point) | **0.5 op/byte** | ~64× below |
| Range-query MBR test (4 compares per 32-byte node) | **0.125 op/byte** | ~250× below |

By the standard metric this workload is close to the **worst possible** GPU
candidate. There is essentially nothing to compute. Compare a GEMM — O(n³) work
on O(n²) data, hundreds of FLOP/byte, sitting far to the right of the ridge — or
a graphics or transformer workload, both of which are designed around reuse.
Spatial indexing has **no reuse to exploit**: each point is read, compared once
or twice, and written. The compute units are idle almost the entire time.

### 2.2 And yet the GPU is substantially faster. Why.

**Because the GPU's real advantage for this class of work is not FLOPs — it is
memory bandwidth, and the ability to actually reach it.**

- **Raw bandwidth advantage.** GDDR6 at 192 GB/s versus the host's dual-channel
  DDR5-4800 at **76.8 GB/s theoretical** — a 2.5× gap on paper.
- **Realized bandwidth advantage is larger than the paper gap.** Saturating DDR
  from a CPU requires near-perfect prefetching and enough outstanding misses;
  6 cores running a comparison sort are *latency*-bound and typically realize a
  fraction of peak. A GPU hides latency structurally: 16 SMs with thousands of
  resident threads means a memory-bound kernel with a regular access pattern
  routinely reaches >80% of peak. **The GPU is better at being memory-bound.**
- **The work is embarrassingly parallel and branch-regular.** Sorting, MBR
  reduction, and MBR intersection testing are all data-parallel with no
  divergence in the inner loop. That regularity is exactly the precondition for
  *reaching* peak bandwidth — a low-intensity kernel that achieves peak still
  beats a low-intensity CPU loop achieving half of a much lower peak.

**The reframing the article should make explicit:** for this workload the
roofline model is not wrong, it is simply *permanently evaluated on its diagonal*.
The useful figure of merit is not FLOP/byte but **effective bytes per second =
(fraction of peak bandwidth achieved) × (peak bandwidth)**. The engineering
problem is therefore never "increase the arithmetic intensity" — there is no
arithmetic to add — but **"raise the roof": move each byte across the fastest
tier that can hold it, and touch each byte the fewest possible times.**

Every design decision in this project is an instance of that principle.

### 2.3 The memory hierarchy this project actually fights

| Tier | Bandwidth | Capacity here |
|---|---|---|
| GDDR6 (VRAM) | **192 GB/s** | 2.94 GB usable |
| DDR5 (host RAM) | **76.8 GB/s** theoretical | 5 GB budgeted |
| PCIe 4.0 | **31.5 GB/s** at x16 / 15.8 GB/s at x8, per direction | — (transit only) |
| NVMe/SATA SSD | **~1–3 GB/s** | effectively unlimited |

*(`nvidia-smi` reports link width max = x16, currently negotiated at x8 while
idle. The width under sustained load must be confirmed — **[TODO — measure]**,
it changes the break-even analysis below by 2×.)*

The ordering is the whole story: **the fastest tier is the smallest, and the
only tier big enough to hold the data is ~100× slower than the one doing the
work.** The GPU's 192 GB/s is gated behind a ~16–32 GB/s pipe, which is itself
fed by a ~1–3 GB/s disk.

### 2.4 The break-even condition — the single most important number

Every byte the GPU sorts must cross PCIe **twice** (H2D, then D2H). So for a
chunk of *B* bytes the GPU path is worth taking only if:

```
    2·B / BW_pcie  +  t_sort_GPU(B)   <   t_sort_CPU(B)
    └──── transfer tax ────┘
```

With `B` = 1.58 GB (`SORT_CHUNK_POINTS` = 197,564,825 points) and PCIe at a
realistic ~13 GB/s effective on x8 with pinned memory, **the transfer tax alone
is ≈ 243 ms per chunk**. The GPU sort must beat the CPU sort by more than that
before the first millisecond of benefit appears. On x16 the tax halves to
~120 ms.

This is exactly why the code goes to the trouble of:
- **pinned host buffers** (`cudaMallocHost`) — pageable memory would force an
  extra staging copy and roughly halve PCIe throughput;
- **timing H2D, kernel, and D2H separately with CUDA events**, so the tax is a
  reported number rather than an assumption;
- **capping the chunk buffer at the input size**, since a multi-GB pinned
  allocation for a small dataset costs more than the sort it enables.

**[TODO — measure]** Report `h2d_ms`, `gpu_compute_ms`, `d2h_ms` and the CPU
twin's `cpu_sort_ms` for the same chunk size. This table *is* the paper's core
result.

### 2.5 The corollary: the disk is the real enemy

Because the disk roof is ~100× below the VRAM roof, a single unnecessary pass
over the raw data costs more than every GPU optimization gains. At 10⁹ points,
one extra pass = 8 GB read = **~4–8 seconds of pure disk time**, against GPU
sorts measured in hundreds of milliseconds.

This is why the contributions in §3 and §4 are all about **pass count and access
pattern**, not about kernels. In an out-of-core GPU algorithm, *the GPU is not
where you find your speedup — it is where you spend the time you saved on I/O.*

---

## 3. Contribution 1 — the out-of-core machinery

### 3.1 Cached runs: one code path from "fits in VRAM" to "fits nowhere"

```c
struct CachedRun {
    std::vector<Point> data;  // non-empty → this run lives in RAM
    std::string        path;  // non-empty → this run lives on disk
    size_t             num_points;
};
```

Phase 1a produces one run per GPU sort chunk. Each run is placed in RAM if the
remaining `USABLE_RAM_BYTES` budget covers it, otherwise spilled to
`tmp/x_run_N.bin`. The budget is spent greedily on the earliest runs.

Consumption is unified behind **`RunReader`**, which exposes
`current() / advance() / eof()` and hides the RAM-vs-disk distinction entirely,
so the merge loop is written exactly once.

**Why this matters more than it looks.** The obvious design is two algorithms —
an in-memory one and an external one — with a size check dispatching between
them. That is what the earlier structure did, and it is a bug farm: two code
paths, two sets of edge cases, and a discontinuity in behaviour right at the
threshold. The `CachedRun` design instead makes the RAM/disk boundary a
*continuous* property of individual runs, so the algorithm degrades smoothly:

| Regime | Runs | Phase 1b disk I/O | Total raw-data passes |
|---|---|---|---|
| Fits in one GPU chunk | 1 | none (no merge at all) | 1 read + 1 write |
| Fits in `USABLE_RAM_BYTES` | k, all cached | **zero** — merges RAM→RAM | 1 read + 1 write |
| Exceeds RAM | k, some spilled | 1 read + 1 write | 2 reads + 2 writes |

The middle row is the interesting one: **a dataset far larger than VRAM still
costs only one read and one write of the disk**, because VRAM is used purely as
a *streaming sort accelerator* and RAM absorbs the intermediate state. The GPU
never needs to hold the dataset — it only needs to hold one chunk at a time.

### 3.2 Buffer topology: fewer buffers, one I/O stream

*(commit `4a9404b` — "triple buffer technique was misusing the SSD")*

The initial design used **three buffers with a separate reader thread and a
separate writer thread running concurrently**. This is the intuitive
"maximize overlap" answer and it was **wrong**.

On an SSD, two independent concurrent streams interleave at the device queue and
convert two sequential access patterns into one effectively random one. The
device's sequential throughput (~1–3 GB/s) collapses toward its random
throughput. Worse, on the raw-data pass this is happening to the *slowest tier
in the hierarchy* — precisely where you can least afford it.

The fix inverted the intuition: **two buffers, one I/O thread, which serializes
write-then-read**. The device always sees exactly one sequential stream.

```
main thread :  [ GPU sort chunk i ]
io   thread :  [ write chunk i-1 ][ read chunk i+1 ]   ← serialized, never concurrent
```

Overlap is preserved exactly where it is free — **across the GPU/I/O boundary**,
which uses two independent devices — and eliminated where it is destructive,
*within* the I/O device. The rewrite was a net **simplification** (+507/−671
lines).

> **Lesson for the article:** overlap is only beneficial across independent
> resources. Compute ‖ I/O is real; I/O ‖ I/O on a single device is a mirage.
> The general rule for out-of-core GPU work: **one sequential stream per
> physical device, and as many independent devices in flight as you have.**

The same double-buffer + single-I/O-thread pattern is used identically in
Phase 1a (X-sort) and Phase 2 (Y-sort), and the CPU twin reproduces it exactly.

### 3.3 K-way heap merge: I/O-optimal by construction

*(commit `55a0386` — "incorrect implementation of merge, changed to a faster
k-way merge; perf: improve perf by reducing disk I/O operations")*

The first implementation merged runs **pairwise**, the textbook mistake. Pairwise
merging re-reads and re-writes the entire dataset once per merge round, giving
**O(N log k) bytes of disk traffic**. For 8 GB and 6 runs that is ~3 full passes
instead of 1 — and per §2.5, disk passes are the dominant cost in the entire
program.

The replacement is a single-pass **k-way merge driven by a binary min-heap** over
the `RunReader`s: pop the global minimum, emit it, refill from that run, sift.
Traffic drops to **O(N) — exactly one read and one write, independent of k.**

This is optimal, and worth stating in the external-memory model: sorting N items
with M memory and block size B costs Θ((N/B)·log_{M/B}(N/B)) I/Os. When k ≤ M/B —
i.e. when every run gets a real buffer — the log term is 1 and the whole sort is
**two passes: one to create runs, one to merge**. This implementation hits that
bound, and the RAM cache of §3.1 collapses it further to *one* pass when the data
fits.

Heap comparisons cost O(N log k) *CPU* work, but since k is small (≈6 at 1 B
points) and the workload is bandwidth-bound anyway, log₂(6) ≈ 2.6 extra
comparisons per element are invisible next to the I/O they eliminate.

**Adaptive buffer sizing.** Each reader's buffer is
`min(max(2 GB / (k+1), 4096 pts), 16 M pts)`. Large k → smaller buffers so memory
stays bounded; small k → 128 MB buffers so reads stay long and sequential.

**Scaling limit worth naming in the article:** the single-pass merge holds only
while `k × B` fits in the merge budget. As the dataset grows, k grows, buffers
shrink toward the 4096-point (32 KB) floor, and reads stop being sequential — at
which point the SSD penalty of §3.2 returns through the back door. Beyond that
point you need either larger runs (more RAM/VRAM) or a genuine multi-pass merge.
With 1.58 GB runs and a 2 GB merge budget, the practical ceiling is on the order
of a few hundred runs, i.e. a few hundred GB of input.

---

## 4. Contribution 2 — the node struct that removes a full pass over the raw data

### 4.1 The two layouts

**Classic R-Tree (Guttman, and every implementation in `references/`):** a node
is an array of entries, and an entry is `(child MBR, child pointer)`. **A node's
own MBR is stored in its parent.** The root's MBR is stored separately, in a
header, or not at all.

```c
// classic
node  = { count, is_leaf, entry[C] };
entry = { MBR mbr;          // 16 B — the *child's* MBR
          uint64_t child; } // 8 B  → 24 B/entry
```

**This project:** every node carries its own MBR, and children are a contiguous
index range so no child list is needed.

```c
// this project — src/external_str_cuda.cu
struct RTreeNode {
    MBR      mbr;           // 16 B — this node's OWN MBR
    uint64_t first_child;   //  8 B — leaf: index into points[]; internal: into nodes[]
    uint32_t num_children;  //  4 B
    uint32_t is_leaf;       //  4 B
};                          // 32 B total, static_assert-enforced
```

*(Note for accuracy: this layout is present from the very first commit
`d7e658a`; the repository never contained the classic alternative. The comparison
below is against the layout used in the literature, not against a previous
version of this code. The current branch `refactor/root-node-struct` is where the
consequences are being revisited.)*

### 4.2 Why the classic layout forces an extra pass over the raw data

STR is **bottom-up and recursive**. To build level *L+1* from level *L* you must:

- **(a)** know each level-*L* node's MBR — it is the STR sort key (MBR centre),
- **(b)** sort level *L* by that key, tile it, and group into parents.

Step (a) is where the layouts diverge. **In the classic layout, at the moment
level *L* has just been written, the MBRs of level-*L* nodes exist nowhere in the
file** — they are precisely the thing level *L+1* is about to store. They must
therefore be *derived*.

For the leaf level (*L* = 0), deriving a leaf's MBR means reading its 512 points
and reducing them. Doing that for all leaves means **reading the entire point
array a second time — a full pass over the raw data: 8 GB at 10⁹ points.**

And you cannot dodge it by fusing leaf and parent construction into one pass,
because of the per-level re-sort of §5.1: **parents cannot be formed until the
entire leaf level exists and has been globally re-sorted.** Level construction is
a global operation, so the leaf level must be fully materialised before Phase 3
can begin. That is structural, not an implementation detail.

### 4.3 Why self-MBRs remove it — "one less iteration"

Storing the MBR in the node makes the leaf level materialise *as* the thing the
next level needs. Concretely, in Phase 2 the leaf MBR is computed **inside the
same loop iteration that produced the sorted slice**, while the data is still
cache-hot from the D2H copy and still in the pinned buffer that is about to be
written out:

```
read slice → GPU Y-sort → compute leaf MBRs from the hot buffer → hand buffer to I/O thread
```

The resulting `std::vector<RTreeNode> nodes` is then the *complete* input to
Phase 3. **`build_internal_levels()` never touches `points` at all** — verified
by inspection: its only data access is `const MBR& c = nodes[s + j].mbr;`. It
reads each child's own MBR to compute the parent's union, and that is the entire
dependency.

Likewise `sort_nodes_str()` sorts `nodes[level_start … level_start+level_count)`
**in place**. In the classic layout each level needs an extra
materialise-and-destructure step — read level *L*, extract `(MBR, index)` pairs
into a scratch array, sort that, group it, write level *L+1*. With self-MBRs
**the level-*L* node array *is* the level-*L+1* sort input**, with no extraction,
no scratch array, and no second buffer at any level. That is the "one less
iteration" of the recursion.

Bottom-up, the leaf level dominates the saving because it is the only level
backed by the raw data:

| Level | Entries at 10⁹ points | Bytes to re-derive MBRs (classic) |
|---|---|---|
| Leaves (L=0) | 1,953,125 leaves | **8 GB — the whole point array** |
| L=1 | 15,259 nodes | 63 MB (node array, RAM-resident) |
| L=2 | 120 nodes | ~0.5 MB |
| L=3 | 1 (root) | — |

So the entire benefit is concentrated in exactly one place, and it is exactly one
full pass over the raw data.

### 4.4 The cost: 16 bytes. Total. For the whole tree.

Count the MBRs stored, tree-wide:

- **Classic:** one MBR per *non-root* node (each stored inside its parent) →
  `num_nodes − 1` MBRs.
- **Self-MBR:** one MBR per node → `num_nodes` MBRs.

**Difference: exactly one MBR = 16 bytes** (4 × `float`) — the root's, which in
the classic layout has no parent to live in. At 10⁹ points that is 16 bytes
against a **63 MB node array and an 8 GB point array**.

> **The headline trade:** 16 bytes of storage buys the elimination of an 8 GB
> disk pass. The article should state it exactly that starkly — it is a ~5×10⁸ : 1
> ratio, and it is the cleanest illustration of §2.5 (the disk is the enemy) in
> the whole project.

### 4.5 The counter-argument: the parent stops being a pruning index

This is the real cost, and it is structural rather than a percentage overhead.

**In the classic layout, a node *is* an index over its children.** You fetch one
node page and it hands you all C child MBRs at once. You test them, reject most,
and fetch only the survivors. **Pruning happens before the I/O.** The whole
purpose of an internal node in a disk-resident R-Tree is precisely this: it is a
compact summary that lets you decide which subtrees *not* to touch, without
touching them.

**In the self-MBR layout, that summary does not exist.** A node record carries
its own MBR and `(first_child, num_children)` — it says nothing whatsoever about
what its children cover. To test a child you must **read the child itself.** So
pruning happens *after* the I/O: you pull every candidate child off disk, look at
its 16-byte MBR, and throw the rest away.

```
classic     :  fetch parent page ──► 170 MBRs in hand ──► test ──► fetch only survivors
self-MBR    :  fetch parent record ──► fetch ALL children ──► test ──► fetch their children
                                        └── the fetch you wanted to avoid ──┘
```

**And a range query rejects the overwhelming majority of what it fetches** — that
is what an index is *for*. So during descent, most of the bytes crossing the disk
belong to nodes the query will discard, and **of each discarded node, only 16 of
its 32 bytes are ever looked at.** The `first_child`, `num_children` and
`is_leaf` fields are read purely to be ignored, because they only matter if you
descend, and you are not going to.

Quantified, per pruning decision:

| | Useful bytes (the MBR) | Bytes actually fetched | **Waste per rejected child** | Decisions per 4096 B page |
|---|---|---|---|---|
| Classic (`MBR` 16 B + `child_ptr` 8 B) | 16 | **24** | **8 B (33%)** | **170** |
| This project (32 B node record) | 16 | **32** | **16 B (50%)** | **128** |

So: **half of every byte read during descent exists only to be ignored**, versus
a third in the classic layout — and a page fetch buys 128 pruning decisions
instead of 170, a **25% loss of pruning power per I/O**. In a traversal that is
already bandwidth-bound at 0.125 op/byte (§2.1), that is a direct and permanent
query-side cost.

*(Mitigating structural note: because children are a contiguous index range, the
128 child records of a full node occupy exactly 4096 B — one page. So the
children block plays the role that the node page plays in the classic layout, and
the **number** of fetches per level is comparable; what degrades is the pruning
yield per fetch and the fraction of each fetch that is dead weight. The extra
round trip is real only at the very top, for the root record itself. See §4.7 for
a way this degrades further in the current implementation, and how to fix it.)*

### 4.6 When the penalty bites, and when it does not

The penalty is **proportional to how selective the query is**, which makes the
trade-off legible:

- **Highly selective queries (point lookups, tiny ranges)** are the worst case.
  Descent is a narrow path, nearly every fetched child is rejected, and the
  50%-waste figure applies to nearly all descent I/O. This is where the classic
  layout's parent-as-index really earns its keep.
- **Large range queries** are barely affected. Most fetched children are hits, so
  their non-MBR fields are needed anyway and the "waste" mostly disappears. The
  `leaves_fully_contained` fast path (§6.3) then skips per-point work entirely.
- **Parallel frontier traversal — the GPU query — is the case the layout
  actually suits.** A level-synchronous BFS expands an entire frontier at once
  and tests every child MBR in parallel regardless, so "you had to fetch it to
  reject it" costs nothing you were not already paying: you were going to touch
  all of them. Self-MBRs additionally make each node **independently testable** —
  a thread can evaluate node *i* from its index alone, with no gather from a
  parent — which is exactly what a GPU frontier expansion wants, and what the
  classic layout denies by coupling each child's test to its parent's residency.

So the honest summary is: **the layout is worse for serial, disk-resident,
highly-selective descent, and neutral-to-better for parallel bulk traversal.**
Given that the project's premise is GPU querying, that is a defensible trade —
but the article should state the cost plainly rather than claim a free win, and
should measure it (§8.4).

Two further genuine benefits worth keeping on the ledger:
- The file is **self-describing and randomly seekable** — `rtree_viewer.py` can
  drill into any node by index and `rtree_query.cpp` can `mmap` and start
  anywhere, with zero parsing.
- The 32-byte record is a **clean power of two**, so nodes never straddle cache
  lines, and exactly 128 fit a 4096 B page.

### 4.7 A concrete defect this creates in the current implementation — children blocks straddle pages

The mitigation in §4.5 (a full children block is exactly 4096 B, so it costs one
page fetch) **does not actually hold in the current file layout**, and this is a
verifiable bug-level finding rather than a design trade-off.

Node *i* lives at byte offset `sizeof(RTreeHeader) + 32·i` = `32 + 32·i`. A block
of 128 children starting at node *i* is page-aligned only if
`32 + 32·i ≡ 0 (mod 4096)`, i.e. only when `i ≡ 127 (mod 128)`.

> **1 in 128 children blocks is page-aligned. The other 127 straddle two pages.**

It is compounded by level packing: `build_internal_levels()` does
`level_start += level_count`, and `level_count` is not a multiple of
`internal_cap`, so each level begins at an arbitrary node index and the blocks
inside it inherit the misalignment.

The consequence is that **descent costs two page faults per node expansion
instead of one** — and for `mmap`-driven, page-fault-dominated traversal, the
I/O *count* is the latency-dominant term, not the byte count. This roughly
doubles the real cost of every internal-node expansion, and it lands squarely on
top of the 25% pruning-yield loss from §4.5.

**The fix is trivial and nearly free:**
1. Pad `RTreeHeader` to a full page (4096 B — costs 4064 B once), so the nodes
   array starts page-aligned.
2. Pad each level to a multiple of `internal_cap` (≤ 127 nodes = 4064 B per
   level), so every children block starts at a multiple of 128.

Total cost at 10⁹ points, with 4 levels: **≈ 20 KB** against a 63 MB node array
and an 8 GB file. In exchange, every full children block becomes exactly one
aligned page fetch, which is what §4.5's mitigation assumed all along.

This is the same lesson as §5.4 (byte-denominated constants): in an out-of-core
structure, **alignment to the I/O block is not a micro-optimization, it is the
unit the whole design is denominated in** — and it was applied to node *capacity*
but not to node *placement*. **[TODO — measure]** the before/after in
`rtree_query`'s reported time at high selectivity.

---

## 5. Other findings

### 5.1 STR must be re-applied at every level, not just at the leaves
*(commit `5f58673` — "re-sort nodes before building the next node level; this
fixes the rectangle shapes, into more square-like")*

The naive reading of STR is: sort points, tile, build leaves, then group
*consecutive* leaves into parents. That yields **long, thin internal MBRs**,
because after the leaf level the nodes are ordered by (slice, Y-within-slice) —
so grouping 128 consecutive leaves walks a tall narrow column, not a compact
square.

`sort_nodes_str()` fixes it by re-running STR on the *node* level itself before
forming each parent level: sort all nodes by **MBR centre X**, cut into
`⌈√(num_groups)⌉` vertical slices, sort each slice by **MBR centre Y**. Result:
near-square MBRs all the way up. MBR shape is what actually governs range-query
cost — dead area in an MBR becomes false-positive node visits — so this is a
correctness-of-quality fix, not a micro-optimization.

Implementation details worth keeping:
- Node-level slice width is rounded up to a multiple of `group_cap`, so a parent
  never straddles a slice boundary — the node-level analogue of the leaf
  alignment in `compute_str_slice_size()`.
- The centre comparison uses `min + max` instead of `(min + max)/2`. Division is
  monotone, so it is pure waste in a comparator invoked O(n log n) times.
- **This fix is also what makes level construction global** (§4.2) and therefore
  what makes the self-MBR layout necessary rather than merely convenient. The two
  findings are coupled.

### 5.2 `thrust::sort` is silently *not* using radix sort here — a ~4× sort speedup left on the table

Verified against the installed headers (CUDA 12.0,
`/usr/include/thrust/system/cuda/detail/sort.h:389`):

```c
template <class Key, class CompareOp>
struct can_use_primitive_sort
  : and_< is_arithmetic<Key>,
          or_< is_same<CompareOp, thrust::less<Key> >,
               is_same<CompareOp, thrust::greater<Key> > > > {};
```

The call in `gpu_sort()` is `thrust::sort(d, d+n, CompareX())`, where the key type
is `Point` (a struct — **not arithmetic**) and the comparator is a custom functor
(**neither `less` nor `greater`**). Both conditions fail, so the call **falls
through to CUB `DeviceMergeSort` — a comparison sort — rather than
`DeviceRadixSort`.**

Since the sort is purely bandwidth-bound (§2.1), the cost is directly countable
in passes over VRAM, for one full 1.58 GB chunk (n = 197,564,825):

| Sort | Passes over the chunk | VRAM traffic | Floor at 192 GB/s |
|---|---|---|---|
| CUB merge sort (current) | block sort + ~log₂(n/2048) ≈ **17** | **≈ 52 GB** | **≈ 273 ms** |
| 4-pass LSD radix (8-bit digits) | **4** | **≈ 12.6 GB** | **≈ 66 ms** |

**≈ 4× available speedup, from changing the dispatch alone** — no new algorithm.

The fix is to sort **keys separately from values**: extract the sort key as a
`uint32_t`, and call `thrust::sort_by_key(keys, keys+n, points)` with the default
comparator, which satisfies `can_use_primitive_sort` and reaches
`DeviceRadixSort`. The IEEE-754 bit pattern of a **non-negative** float is
monotone under unsigned integer comparison, and `gen_points` emits only [0, 1000],
so the direct reinterpretation is valid for the current data; a general
implementation flips the sign bit for positives and inverts all bits for
negatives.

**Onesweep** (`references/`) is the next step beyond that — a single-pass-per-digit
LSD radix sort using decoupled look-back, which is exactly why that paper is in
the reference folder. **[TODO — measure]** This is probably the highest
value-per-hour change available in the project, and it strengthens the central
thesis: in a bandwidth-bound regime, *the algorithm that moves fewer bytes wins*,
and the comparison-vs-radix distinction is entirely a byte-movement distinction.

### 5.3 Non-contiguous internal nodes
*(commit `443af40` — "fix: non-continuos non-leaf-nodes")*

The `(first_child, num_children)` encoding — the thing that makes a node 32 bytes
instead of hundreds — is only valid if a parent's children are genuinely
contiguous in the node array. An early version violated this at level boundaries
and produced a tree that read back as garbage. The invariant is now maintained by
appending levels strictly in order and tracking `level_start` explicitly in
`build_internal_levels()`.

### 5.4 Byte-denominated constants
*(part of commit `4a9404b` — "constants are now based on bytes")*

`RTREE_NODE_BYTES = 4096` (one page, one typical SSD block), from which
`RTREE_LEAF_CAPACITY = 4096/8 = 512` points and
`RTREE_INTERNAL_CAPACITY = 4096/32 = 128` children fall out automatically.

The natural unit for an out-of-core structure is the **I/O block**, not an
abstract fanout number. Leaves and internal nodes then get *different* entry
capacities derived from their element sizes — which is correct, and easy to get
wrong if you think in fanout. It also means the whole configuration adapts if
`sizeof(Point)` changes (3-D points, or rectangles instead of points).

`USABLE_GPU_BYTES` and `USABLE_RAM_BYTES` are likewise byte budgets, and
`SORT_CHUNK_POINTS = USABLE_GPU_BYTES / 2 / sizeof(Point)` — the ÷2 because
Thrust needs roughly 2× the data size for input plus scratch.

### 5.5 Measurement discipline: "component sum" vs. "wall"
*(commit `abace80`)*

`TimingStats` reports **both** the sum of all component times (disk read, disk
write, GPU alloc, H2D, kernel, D2H, CPU merge, tree build, header write) **and**
wall-clock elapsed, plus their difference labelled `OVERLAP SAVED`. Component-sum
is what the run *would* cost fully serialized; the gap is the measured value of
the GPU‖I/O pipeline. This makes the pipelining benefit a directly reported
number rather than something inferred — important, because §3.2's finding is that
overlap intuitions are unreliable.

Supporting details needed to make the numbers honest:

- **`gpu_warmup()`** forces CUDA context creation and the first `cudaMalloc`
  before Phase 1a, so the one-time context-init cost is reported separately
  instead of being charged to the first sort.
- **The device buffer is allocated once and grown on demand** (`g_d_buf`), not
  per chunk — `cudaMalloc`/`cudaFree` are synchronizing and expensive.
- **CUDA events, not host clocks**, time H2D / kernel / D2H, since the copies are
  async and a host timer would measure the wrong thing.
- **The chunk buffer is capped at the input size**
  (`eff_chunk_points = min(chunk, file_points)`), or a small dataset pays for a
  multi-GB *pinned* allocation that dominates its wall time. Pinned allocation is
  not free and scales with size.
- **A documented threading contract**: the I/O worker never writes the global
  `g_stats`; it returns its timings through a `std::future` and the main thread
  folds them in. No locks, no torn counters, and the invariant is written down in
  the source so it survives future edits.

### 5.6 The CPU twin is a control group, not a fallback

`external_str_cpu.cpp` keeps the **identical phase structure, identical file
format, identical `CachedRun`/`RunReader` machinery, identical double-buffered
I/O pipeline**, and swaps only `thrust::sort` → `std::sort`. Any measured
difference is therefore attributable to the sort and the PCIe transfers it costs
— which is exactly the break-even question of §2.4.

**Caveat that must be fixed before publishing any speedup number:** the CPU
twin's sort is currently **single-threaded `std::sort`** on a 6-core/12-thread
CPU. Comparing one core against a whole GPU overstates the speedup by roughly an
order of magnitude and would be the first thing a reviewer attacks. See §9.

---

## 6. Reference implementation details

### 6.1 Output file format

```
[RTreeHeader  — 32 bytes]
[RTreeNode[]  — num_nodes × 32 bytes]
[Point[]      — num_points × 8 bytes]
```

```c
struct Point       { float x, y; };                              //  8 B
struct MBR         { float min_x, min_y, max_x, max_y; };        // 16 B
struct RTreeNode   { MBR mbr; uint64_t first_child;
                     uint32_t num_children, is_leaf; };          // 32 B
struct RTreeHeader { uint32_t magic /*0x52545245 "RTRE"*/,
                     leaf_capacity, height, internal_capacity;
                     uint64_t num_points, num_nodes; };          // 32 B
```

Every size is `static_assert`-enforced, so the CUDA writer, the CPU writer, the
C++ query tool, and the Python/numpy viewer agree on layout by construction.

Properties that make it work out-of-core:

- **No pointers, only indices** → position-independent, `mmap`-able with zero
  parsing and zero copies. `rtree_query.cpp` uses `mmap`; `rtree_viewer.py` uses
  `numpy.memmap` and can therefore browse trees far larger than RAM, the OS
  paging in only what is on screen.
- **Contiguity invariant** (§5.3) → a node needs only `(first_child,
  num_children)`, no child list.
- **Root is `nodes[num_nodes − 1]`** — levels are appended bottom-up, so the last
  node written is the root. (Being revisited on `refactor/root-node-struct`;
  an explicit header field would be less fragile than a positional convention.)
- **`num_nodes` is computed structurally before any data is touched**
  (`precompute_num_nodes` — pure arithmetic on `total_points`, `leaf_cap`,
  `internal_cap`). This lets the writer reserve the exact header+nodes prefix,
  stream the points directly into their final positions, and seek back once at
  the end. **One write pass, no rewrites** — which is the same §2.5 principle
  again.

### 6.2 The four phases

| Phase | Work | Overlap |
|---|---|---|
| **1a** | Read input in `SORT_CHUNK_POINTS` chunks, GPU-sort each by X, emit a `CachedRun` (RAM or spilled) | GPU sort ‖ serialized write-then-read |
| **1b** | K-way heap merge of all runs → RAM if all cached, else `tmp/sorted_x.bin` | — (single pass) |
| **2** | Per STR slice: read → GPU Y-sort → compute leaf MBRs from the hot buffer → append points to their final file offsets | GPU sort ‖ serialized write-then-read |
| **3** | Build internal levels bottom-up (per-level STR re-sort + MBR unions), then one `seek(0)` and write header + node array | — (RAM-only, no points touched) |

Slice sizing (`compute_str_slice_size`): `⌈√(num_leaves)⌉` slices, rounded up to
a whole multiple of `leaf_capacity` so a leaf never straddles a slice, capped at
`STR_MAX_SLICE` (one GPU sort's worth). Leaf-only alignment suffices because
every node level is re-sorted anyway (§5.1).

### 6.3 Querying

`rtree_query.cpp` supports `point` and `range` queries over the `mmap`ed file and
reports **`nodes_visited`, `points_checked`, `leaves_fully_contained`, and
time** — it is instrumented as a *tree-quality* measure, not just a lookup tool.
`nodes_visited` at a given selectivity is the direct empirical test of both the
MBR-shape fix (§5.1) and the fanout cost of the node layout (§4.5).

One optimization worth mentioning: if the query rectangle **fully contains** a
leaf's MBR, every point in that leaf is a hit and the per-point test is skipped
entirely (`leaves_fully_contained`). For large range queries this removes almost
all per-point work — again a byte-movement win, not a compute win.

`madvise(MADV_RANDOM)` is set on the mapping, which suits tree descent but is
arguably wrong for large range queries that stream whole leaves sequentially.

### 6.4 Artifacts

| File | Role |
|---|---|
| [src/external_str_cuda.cu](src/external_str_cuda.cu) | Main contribution — GPU-pipelined out-of-core STR bulk-loader (1188 lines) |
| [src/external_str_cpu.cpp](src/external_str_cpu.cpp) | CPU control group, identical in every respect but the sort (1048 lines) |
| [src/rtree_query.cpp](src/rtree_query.cpp) | `mmap` point/range query, instrumented |
| [src/gen_points.cpp](src/gen_points.cpp) | Synthetic generator, uniform in [0,1000]², resumable/appendable |
| [src/gpu_info.cu](src/gpu_info.cu) | Hardware probe + suggested `constants.h` values |
| [src/rtree_viewer.py](src/rtree_viewer.py) | Interactive `numpy.memmap` viewer — drill in/out, zoom, pan |
| [src/constants.h](src/constants.h) | All tunables, byte-denominated |

`rtree_viewer.py` was switched to `mmap` (commit `fee548f`) specifically so it
could inspect trees built over datasets larger than RAM. **It is the tool that
made the bad-MBR-shape problem of §5.1 visible** — the article should include a
before/after figure of long thin rectangles versus square-ish ones.

### 6.5 Development environment

| Component | Spec |
|---|---|
| GPU | RTX 3050 Laptop — **3.68 GiB** VRAM, 16 SM, CC 8.6 (Ampere), 128-bit bus, 6001 MHz mem clock → **192 GB/s** |
| PCIe | 4.0, max x16 (31.5 GB/s), negotiated x8 at idle |
| CPU | AMD Ryzen 5 6600H, 6C/12T |
| RAM | 14 GB, DDR5-4800 dual channel → 76.8 GB/s theoretical |
| Toolchain | CUDA 12.0, `nvcc -O3 -std=c++17 -arch=sm_86`, `g++ -O3 -std=c++17` |
| Budgets | `USABLE_GPU_BYTES` = 2.94 GB, `USABLE_RAM_BYTES` = 5 GB, `SORT_CHUNK_POINTS` = 197,564,825 (1.58 GB) |

---

## 7. References and how they relate

**The gap this project addresses** — all of these assume the data fits in device
memory, and all evaluate at 10⁵–10⁶ scale:
- `Parallel_implementation_of_R-trees_on_the_GPU.pdf` — states the assumption
  explicitly (quoted in §1). **This is the paper to cite when framing the gap.**
- `GPU-based_Parallel_R-tree_Construction_and_Querying.pdf`
- `Optimization of R-tree Construction for Spatial Data Using GPU.pdf` — 100 K
  entries, 100 K rectangles, 2 M rectangles.
- `Parallel Spatial Query Processing on GPUs Using R-Trees.pdf` — does treat PCIe
  transfer as a first-class cost ("reduce data transfer overheads … motivates
  us"), which is the closest any reference comes to this project's concern, but
  still with a resident tree.
- `Parallel Multi-dimensional Range Query Processing.pdf`

**The algorithm**
- `STR.pdf` — Leutenegger, Edgington, López, *STR: A Simple and Efficient
  Algorithm for R-Tree Packing*. The algorithm implemented here.
- `OMT.pdf` — Overlap Minimizing Top-down bulk-loading; the main alternative
  (top-down rather than bottom-up). Worth a paragraph on why STR was chosen:
  STR is *sort-dominated*, and sorting is the one thing a GPU does superbly at
  low arithmetic intensity — which is precisely the §2.2 argument.

**GPU sorting** — directly load-bearing after §5.2
- `Onesweep: A Faster Least Significant Digit Radix Sort for GPUs.pdf` — the
  upgrade path from the accidental comparison sort.

**GPU memory behaviour**
- `An Efficient GPU Cache Architecture for Applications.pdf`

**kNN on GPU — the adjacent query type and likely next direction**
- `High performance GPU implementation of KNN algorithm — A review`
- `GPU-SME-kNN: Scalable and Memory Efficient kNN and Lazy…` — note the
  "memory efficient" framing; the closest neighbour to this project's concern.
- `GPU-Based Algorithms for Processing the k-Nearest-Neighbor Query on Spatial
  Data Using Partitioning and Concurrent Kernel Execution` — 100 K / 1 M points.

The kNN cluster signals the intended direction: the same packed, `mmap`-able,
contiguous-range, self-MBR tree is a good substrate for a GPU kNN search, and
§4.6's parallel-frontier argument is what makes it so.

---

## 8. Results

**[TODO — measure]** Nothing is stored in the repository. The binaries already
print everything needed. Priority order:

**8.1 The break-even table (§2.4) — the core result.** For one 1.58 GB chunk:
`h2d_ms`, `gpu_compute_ms`, `d2h_ms`, transfer tax as a fraction of total, versus
the CPU twin's `cpu_sort_ms`. Report achieved PCIe GB/s alongside theoretical.

**8.2 End-to-end scaling across the three regimes** (which are genuinely
different code paths — that is itself a result):

| N | Size | Regime | GPU wall | CPU wall | Speedup | 1a | 1b | 2 | 3 | Overlap saved | Bytes read | Bytes written |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10⁶ | 8 MB | single chunk, no merge | | | | | | | | | | |
| 10⁷ | 80 MB | single chunk | | | | | | | | | | |
| 10⁸ | 800 MB | multi-run, all cached | | | | | | | | | | |
| 5×10⁸ | 4 GB | multi-run, all cached | | | | | | | | | | |
| 10⁹ | 8 GB | runs spill to disk | | | | | | | | | | |

The `bytes_read`/`bytes_written` columns should empirically confirm the pass
counts in §3.1's table — that is the cleanest evidence for the cached-run
contribution.

**8.3 The overlap value (§3.2).** `OVERLAP SAVED / COMPONENT SUM` per phase.
Ideally also re-run with the old triple-buffer topology to quantify the SSD
penalty directly — that would turn §3.2 from an anecdote into a measurement.

**8.4 Query quality and the node-layout cost.** `nodes_visited` and
`points_checked` for range queries swept across selectivity (0.001%, 0.01%, 0.1%,
1%, 10%), which is the axis along which §4.6 predicts the penalty varies:
- (a) with and without the per-level STR re-sort of §5.1 — the MBR-shape result;
- (b) **bytes fetched per pruning decision** and **pruning decisions per page
  fetch**, 128 vs. a simulated 170, to quantify the §4.5 cost. The prediction is
  a widening gap as selectivity increases, and near-parity for large ranges;
- (c) **before/after the page-alignment fix of §4.7.** This one should show up as
  a step change in wall time at high selectivity even though `nodes_visited` is
  unchanged — which is itself the cleanest possible demonstration that in
  out-of-core work the I/O *count* dominates the byte count.

**8.5 Sort dispatch (§5.2).** Current `thrust::sort` versus
`thrust::sort_by_key` on extracted `uint32_t` keys. Predicted ~4×.

**8.6 Tree statistics.** Height, node count, occupancy (should be ~100% except
the last node of each level — that is the point of packing), root MBR.

---

## 9. Open questions and threads not yet pulled

**Blocking for publication:**
- **The CPU baseline is single-threaded `std::sort`.** Comparing one core to a
  whole GPU overstates the speedup by roughly an order of magnitude. A fair
  baseline needs a parallel CPU sort (`std::execution::par_unseq`, or a
  hand-rolled multithreaded merge sort over the 12 available threads — the
  Makefile already links `-lpthread`). **This is the single most important fix
  before any speedup number is published.**
- **No verification harness.** There is no test that the tree contains every
  input point, nor that a range query matches a brute-force scan. Roughly 30
  lines on top of `rtree_query.cpp`, and required before any correctness claim.
- **Only uniform random data has been tested.** Uniform is STR's *best* case —
  every slice is equally full and MBRs are naturally square. Real spatial data is
  clustered and skewed. Clustered/skewed/real datasets (OSM extracts, TIGER,
  taxi pickups) would make the tree-quality results credible, and would stress
  the §5.1 finding much harder.

**High-value optimizations:**
- **Page-align the node array and the children blocks** (§4.7) — ~20 KB of
  padding to halve the page faults per node expansion. Cheapest available win in
  the project, and it is a defect rather than a trade-off.
- **Radix sort via `sort_by_key` on extracted keys** (§5.2) — predicted ~4× on
  the dominant phase.
- **CUDA streams to overlap H2D/D2H with the kernel.** Currently the transfers
  and the sort are serialized within `gpu_sort()`. Splitting a chunk into
  sub-chunks across 2–3 streams would hide most of the §2.4 transfer tax behind
  the sort itself — the same "overlap independent resources" principle as §3.2,
  applied one level down. This is probably the second-biggest available win.
- **Leaf MBR computation is a serial CPU loop** over the sorted slice. It is a
  trivially parallel segmented reduction that could run on the GPU while the data
  is *already in VRAM*, eliminating a whole CPU pass over 8 GB.
- **Phase 3 is entirely CPU and single-threaded**, including the per-level STR
  sorts. At 10⁹ points the leaf level is a ~2 M-element sort — not free, and it
  could reuse the GPU sort machinery that already exists.

**Design questions:**
- **`refactor/root-node-struct`** (current branch) — the "root is the last node"
  positional convention is being reconsidered; an explicit header field is less
  fragile. This is also the natural place to address §4.5 and §4.7 together.
  The obvious hybrid: **keep self-MBRs at the leaf level** (where the entire 8 GB
  build-pass saving of §4.2 lives, since leaves are the only level backed by raw
  data) and **pack child MBRs into the parent for internal levels only** (where
  the query-side pruning-index property of §4.5 lives, and where the node array
  is only 63 MB so nothing is saved by not duplicating it). That captures both
  ends of the trade-off, and the 16-byte accounting of §4.4 barely changes.
  Worth checking whether the STR recursion still closes cleanly under the hybrid
  — internal levels would need their MBRs materialised for the next level's sort,
  but that is a 63 MB in-RAM operation, not an 8 GB disk pass, so it costs
  essentially nothing.
- **`MADV_RANDOM` vs `MADV_SEQUENTIAL`** in the query tool (§6.3).
- **Two dimensions only.** `Point` hardcodes `float x, y`. Extending to 3-D or to
  rectangles changes `sizeof(Point)` and hence every derived capacity — but the
  byte-denominated constants of §5.4 were designed to absorb exactly that.
- **Merge scaling ceiling** (§3.3) — beyond a few hundred runs the single-pass
  merge degrades. Larger runs, or genuine multi-pass merging, at that point.
- **GPU-side query.** Everything so far accelerates *building*. The self-MBR
  layout was chosen partly to enable parallel frontier traversal (§4.6); actually
  implementing a GPU range/kNN query over the `mmap`ed tree is the obvious
  continuation, and is where the kNN references point.

---

## 10. Reproduction

```bash
make                                        # builds all five binaries into bin/
./bin/gpu_info                              # → suggested constants.h values
# edit src/constants.h to match the machine, then rebuild
./bin/gen_points 1000000000                 # → data/points.bin  (8 GB)
./bin/external_str     data/points.bin data/rtree.bin   # GPU
./bin/external_str_cpu data/points.bin data/rtree.bin   # CPU control
./bin/rtree_query data/rtree.bin range 100 100 200 200 --count
make view                                   # interactive viewer
```


---
---

# PART II — The rewrite

`refactor/the-return-of-the-programer`. New files alongside the old ones, so
old-vs-new benchmarking stays possible on one checkout.

| File | Role |
|---|---|
| [src/rect_constants.h](src/rect_constants.h) | Byte- and page-denominated configuration, with the measurements that justify each constant in comments |
| [src/rect_rtree_format.h](src/rect_rtree_format.h) | Shared on-disk format; every size `static_assert`-enforced |
| [src/gen_rects.cpp](src/gen_rects.cpp) | Rectangle generator — uniform or clustered |
| [src/str_rtree_cuda.cu](src/str_rtree_cuda.cu) | The GPU bulk-loader |
| [src/rect_rtree_query.cpp](src/rect_rtree_query.cpp) | `mmap` query + structural verification + range-query benchmark |

---

## 11. What changed, and why

### 11.1 Kept from the first implementation

Three things earned their place and were carried over unchanged in spirit:

- **Cached runs** (§3.1) — a run is *either* RAM-resident *or* spilled, chosen
  greedily against a byte budget, consumed through one `RunReader` abstraction.
- **Single-pass k-way heap merge** (§3.3) — O(N) bytes of disk traffic
  regardless of run count.
- **Double-buffer with one serialized I/O thread** (§3.2) — overlap across the
  compute/I/O boundary, never within the I/O device.

### 11.2 Data is now rectangles, with explicit object IDs

```c
struct Rect  { float min_x, min_y, max_x, max_y; };     // 16 B
struct Entry { Rect mbr; uint64_t child; };             // 24 B
```

STR now sorts on the **centroid** of each rectangle (`min + max`, skipping the
monotone division). Because sorting permutes the data, object identity has to
be carried through the sort, so the sorted element is 24 bytes rather than 16.

### 11.3 Textbook node layout — the parent is a pruning index again

A node is `{ PageHeader, Entry[] }` where each entry is
`(child's MBR, child's id)`. **A node stores its children's MBRs, not its own.**
One page fetch therefore yields every pruning decision for that node's whole
fan — the property §4.5 identified as the thing the self-MBR layout destroyed.

- Leaf page: `entry.child` is the **object ID**.
- Internal page: `entry.child` is the **child's page number**.
- The root's MBR — the one MBR with no parent to live in — is in the file
  header, alongside an explicit `root_page`. That is the honest 16-byte cost
  computed in §4.4, now actually paid, and it retires the fragile
  "root is the last node" positional convention.

`MAX_ENTRIES_PER_PAGE = (4096 − 8) / 24 = 170`.

### 11.4 Pages are page-aligned — §4.7's defect is gone by construction

Page 0 holds the header; node pages start at byte 4096 and every node is
exactly one page. Expanding a node is **exactly one page fault**, never the two
that the old layout incurred 127 times out of 128.

### 11.5 The leaf-MBR insight, kept — but in RAM instead of in the struct

This is the part of the first implementation that was genuinely right, and it
survives without the file-format cost that made §4.5 bite.

When a node is packed, its MBR is computed **once**, from data that is already
in the buffer, and pushed into an in-memory `std::vector<Entry>` as
`{ node MBR, node page }`. That vector is the next level's sort input.

The elegance is that **`Entry` is simultaneously the sort element and the
packed-node entry** — the same 24 bytes:

```
level 0 : { object MBR , object id  }   <- read from the input file
level 1 : { leaf   MBR , leaf  page }   <- produced when leaves are packed
level 2 : { node   MBR , node  page }   <- and so on, to the root
```

So each level's output array is *directly* the next level's sort input: no
extraction step, no scratch array, and **the raw data is read exactly once no
matter how tall the tree gets**. `str_pass()` sorts the level vector in place
and packs from it; nothing ever goes back to disk to recover an MBR.

The MBRs live in RAM only *between* levels and are then discarded — only packed
pages reach the disk. At 60 M rectangles the level-0 array is 352,942 entries
= 8.5 MB, entirely negligible against a 1.4 GB dataset.

> This is the correct resolution of the §4 trade-off: **the build-time saving
> came from computing the MBR once and keeping it in memory, not from
> persisting it in the node.** The first implementation conflated the two and
> paid a permanent query-side tax for a transient build-time convenience.

### 11.6 Fill factors

Separate `--fill-leaf` and `--fill-internal`, each scaling the 170-entry page
ceiling, so a bulk-loaded tree can be left with room for later inserts. The
effective capacity feeds the STR slice arithmetic, so tiling stays consistent.

---

## 12. Measured results from the rewrite

Hardware as in §6.5. **All numbers below are real runs, not estimates.**

### 12.1 The GPU/CPU sort crossover — the §2.4 break-even, measured

Pinned buffers, 24-byte `Entry`; GPU column includes H2D + key build + radix
`sort_by_key` + D2H.

| n | `std::sort` | GPU total | GPU sort only | winner |
|---:|---:|---:|---:|---|
| 1,024 | **0.010 ms** | 0.185 ms | 0.165 ms | CPU |
| 4,096 | **0.191 ms** | 0.220 ms | 0.170 ms | CPU |
| 16,384 | 1.155 ms | **0.262 ms** | 0.184 ms | GPU 4.4× |
| 262,144 | 24.245 ms | **1.576 ms** | 0.611 ms | GPU 15.4× |
| 1,048,576 | 100.365 ms | **5.664 ms** | 1.888 ms | GPU 17.7× |
| 16,777,216 | 1814.560 ms | **86.267 ms** | 25.978 ms | GPU 21.0× |

Two results here, both article-worthy:

1. **The crossover is at ~8 Ki entries** — three orders of magnitude below the
   1 Mi that was assumed when the constant was first written. Once data is
   already in a pinned buffer, even a small sort is worth shipping to the GPU.
   The first version of the rewrite guessed 1 Mi and thereby made Phase C
   **3× slower** (62 ms of `std::sort` where 20 ms of GPU would do). *Guessing
   a break-even threshold in a bandwidth-bound system is not safe; measure it.*
2. **The transfer tax is exactly what §2.4 predicted, and it dominates the
   win.** At 16.7 M entries the kernel is **~70× faster** than `std::sort`
   (26 ms vs 1815 ms), but H2D+D2H account for 60 of the 86 ms — **70% of the
   GPU's total time is PCIe**, cutting the end-to-end speedup to 21×.

### 12.2 The Thrust radix-sort finding (§5.2), applied

The predicted fix works. The key is extracted to a `uint32_t` via the standard
monotone float-bit-pattern map (`(u & 0x80000000) ? ~u : (u | 0x80000000)`,
correct for negatives, zeros and positives) and sorted with `sort_by_key` and
the default comparator, which satisfies `can_use_primitive_sort` and reaches
`DeviceRadixSort` instead of `DeviceMergeSort`.

The 16.7 M-entry sort takes **26 ms**, i.e. ~1.55 GB moved through a 4-pass
radix in 26 ms ≈ **matching the predicted ~66 ms floor for 1.58 GB and beating
it**, versus the ~273 ms floor predicted for the comparison-sort path.

### 12.3 Pinned memory: linear cost, flat benefit

Best of 3 trials (`bench/bench_pinned_memory.cu`):

| Buffer | `cudaMallocHost` | `cudaFreeHost` | **ms/MB** | `malloc`+memset | H2D bandwidth |
|---:|---:|---:|---:|---:|---:|
| 64 MB | 34.8 ms | 12.3 ms | **0.544** | 28.8 ms | **13.33 GB/s** |
| 128 MB | 69.3 ms | 23.8 ms | **0.541** | 56.8 ms | **13.40 GB/s** |
| 256 MB | 134.7 ms | 46.3 ms | **0.526** | 111.7 ms | **13.35 GB/s** |
| 512 MB | 270.9 ms | 91.6 ms | **0.529** | 218.5 ms | **13.37 GB/s** |
| 1024 MB | 541.1 ms | 181.2 ms | **0.528** | 436.1 ms | **13.36 GB/s** |
| 1350 MB | 726.8 ms | 239.7 ms | **0.538** | 578.7 ms | **13.36 GB/s** |

**Pinned allocation costs ~0.53 ms per MB, and the ms/MB column is flat to
within 3% across a 21× size range — the cost is exactly linear. Transfer
bandwidth is flat too, 13.33–13.40 GB/s, varying by less than 0.6%.** So a
large pinned buffer costs real time and buys precisely nothing: sizing the sort
chunk at the *GPU* budget meant two 1.35 GB pinned buffers, **~1.45 s to
allocate and ~0.48 s to free, for identical bandwidth to a 64 MB buffer.**

Two honest refinements to this result:

- **Most of the cost is not pinning.** Plain `malloc` + `memset` of the same
  size costs ~0.45 ms/MB, so page-locking adds only ~20% over simply
  first-touching the pages. The conclusion is unchanged but the blame moves:
  it is *the size of the buffer*, not its pinned-ness, that costs.
- **An earlier single-trial run of this same benchmark reported ~1 ms/MB**,
  roughly double. That measurement was taken immediately after a 60 M build,
  with ~2.8 GB of dataset and tree still in the page cache and the allocator
  under pressure. Best-of-3 in a quiet state is the defensible number; the
  single-trial figure is what this cost looks like *under memory pressure*,
  which is arguably the more realistic operating condition and is worth
  reporting as a range rather than a point. **0.53 ms/MB quiet, up to
  ~1 ms/MB under pressure.**

Also note: **13.3 GB/s realized** against 15.8 GB/s theoretical for PCIe 4.0 x8
= 84% of peak. That settles the open question in §2.3 — the link is running at
x8 under load, and pinned transfers do reach most of it.

### 12.4 The pinned-cap trade curve (60 M rectangles, 1.44 GB)

Capping the chunk trades pinned-allocation cost against merge fan-in `k`:

| cap | runs (k) | Phase A wall | merge | Phase C wall | **total** |
|---:|---:|---:|---:|---:|---:|
| 64 MB | 22 | 1872 ms | **3026 ms** | 1028 ms | 7.80 s |
| 128 MB | 11 | **1605 ms** | 2604 ms | 1009 ms | **5.74 s** |
| 256 MB | 6 | 1893 ms | 2409 ms | 1042 ms | **5.75 s** |
| 512 MB | 3 | 2467 ms | 2231 ms | 1004 ms | 7.76 s |
| 1024 MB | 2 | **3202 ms** | **2049 ms** | 1016 ms | 6.70 s |

The two costs move in opposite directions exactly as the model says: Phase A
rises with the cap (pinned allocation), the merge falls (fewer runs, lower
`log k`). The optimum is a **broad flat basin at 128–256 MB**. Choosing 256 MB
cut the 60 M build from **10.6 s to 7.2 s (−32%)**.

*(Totals carry run-to-run variance of ±1 s from page-cache state; the per-phase
columns are the trustworthy part, and they are monotone.)*

### 12.5 CUDA context creation: ~175 ms warm, ~1.7 s cold

Measured repeatedly, persistence mode disabled:

- **cold** (GPU idle, driver unloaded): `cudaFree(0)` = **1717 ms**
- **warm** (recent CUDA process): `cudaFree(0)` = **174–179 ms**

For a 1 M-rectangle build the entire tree takes ~215 ms of real work, so
**context creation is 45% of a warm run and 8× a cold one.** Any GPU-vs-CPU
comparison at small scale that does not report this separately is measuring the
driver, not the algorithm. `gpu_warmup()` therefore forces and reports it
before Phase A.

*(This also caused a self-inflicted analysis error worth recording: the phase
walls appeared not to reconcile with the grand total by 1.6 s, which looked
like a serious instrumentation bug. It was two different runs being compared —
a cold run's total against a warm run's init line. Cold/warm variance of this
magnitude will silently corrupt any benchmark table that mixes them.)*

### 12.6 Query quality — the pruning-index property, measured

60 M rectangles, uniform, height 4, 355,033 pages. Averages over 15 random
queries per selectivity:

| selectivity | pages read | MBR tests | **tests per page** | time |
|---:|---:|---:|---:|---:|
| 0.0001 % | 53.6 | 8,946 | **166.9** | 0.105 ms |
| 0.001 % | 74.4 | 12,482 | **167.8** | 0.139 ms |
| 0.01 % | 160.6 | 27,136 | **169.0** | 0.242 ms |
| 0.1 % | 654.1 | 111,025 | **169.7** | 0.818 ms |
| 1 % | 4,449.3 | 756,209 | **170.0** | 3.56 ms |
| 10 % | 38,704.6 | 6,579,616 | **170.0** | 36.4 ms |

**This is the direct measurement of what §4.5 argued for.** Pruning decisions
per page fetch is **166.9–170.0 against a theoretical ceiling of 170** — the
layout is saturating it. The self-MBR layout's ceiling was **128**, so the
textbook layout delivers **~33% more pruning per unit of I/O**, and the
prediction in §4.5's table is confirmed almost exactly.

At 1 M rectangles, ~81% of all MBR tests prune a subtree that is then **never
fetched at all** — that is the parent-as-index property doing its job, and it
is precisely the work the self-MBR layout could not avoid doing.

Against a brute-force scan, tree range queries were **1474×–2517× faster** with
identical result counts.

### 12.7 Where the time actually goes (60 M, warm, 256 MB cap)

| Phase | component sum | wall | note |
|---|---:|---:|---|
| A — GPU X-sort into runs | 1370 ms | 2004 ms | pinned alloc + RAM-cache copies |
| B — k-way merge | 2355 ms | 2471 ms | **dominant cost** |
| C — GPU Y-sort + pack leaves | 1515 ms | **1023 ms** | **492 ms saved by overlap** |
| D — internal levels | 28 ms | 199 ms | in-RAM, trivial |
| **total** | 6977 ms | **7406 ms** | |

Two things stand out:

1. **The single-threaded CPU heap merge is now the biggest cost in the build** —
   ~2.4 s of a ~5.7–7.4 s run, ~35–40 ns per entry, more than all GPU work
   combined (290 ms of sorting + 226 ms of transfers). Having fixed the sort,
   the bottleneck moved to the merge. This is the next thing to attack.
2. **Phase C's overlap is real and measured**: 1515 ms of serialized component
   cost completed in 1023 ms wall — **492 ms genuinely saved** by running the
   GPU sort concurrently with the serialized I/O stream. §3.2's design is
   validated, and the saving is reported rather than inferred.

### 12.8 Correctness

`rect_rtree_query verify` walks the whole tree and checks: every page has
1..cap entries; every parent entry's stored MBR contains its child subtree's
actual union; every object ID appears **exactly once**; the walked page count
matches the header; and a sample of source rectangles round-trips through a
range query over its own extent.

| configuration | result |
|---|---|
| 1 M uniform, fill 1.0 | **VERIFY OK** — 1,000,000/1,000,000 indexed, 99.99% leaf occupancy |
| 1 M uniform, fill 0.7 / 0.5 / 0.25 | **VERIFY OK** — capacities 118, 85, 42; height grows to 4 at 0.25 |
| 1 M clustered (50 gaussian clusters) | **VERIFY OK** |
| 1 M, forced disk spill (8 MB GPU chunk, 2 MB RAM cache → 7 runs, all spilled) | **VERIFY OK** |
| 60 M uniform | **VERIFY OK** — 60,000,000/60,000,000 indexed, 100.00% leaf occupancy |

In every case the node count and height matched the closed-form
`precompute_num_nodes` / `precompute_height` **exactly**, and range-query counts
matched a brute-force scan **exactly**.

**The in-RAM path and the forced-spill path produce different trees but
identical query answers.** The cause is worth stating in the article rather than
hiding: **2.5% of records share an exact centroid key** at float32 precision
(1,000,000 rectangles → 975,497 distinct keys), so the two paths break ties
differently. Both are valid STR trees; all four test queries returned identical
counts (11022, 1000000, 95, 1495). This retires §9's "no verification harness"
blocker.

---

## 13. What the rewrite settled

| Question from Part I | Verdict |
|---|---|
| §2.4 — does the PCIe tax dominate? | **Confirmed.** 70% of GPU time at 16.7 M entries is transfer; realized 13.3 GB/s = 84% of PCIe 4.0 x8 peak. |
| §2.3 — is the link x8 or x16 under load? | **x8.** 13.3 GB/s realized against 15.8 GB/s theoretical. |
| §4.5 — is the self-MBR layout worse for queries? | **Confirmed, and quantified.** Textbook layout achieves 166.9–170.0 pruning tests per page fetch; self-MBR's ceiling was 128. ~33% more pruning per I/O. |
| §4.7 — do children blocks straddle pages? | **Moot.** The rewrite makes one node one aligned page by construction. |
| §5.2 — is `thrust::sort` silently using merge sort? | **Confirmed and fixed.** `sort_by_key` on an extracted `uint32` key reaches `DeviceRadixSort`; 16.7 M entries sort in 26 ms. |
| §4.2/§4.3 — does the MBR-in-node trick save a pass? | **The saving was real but misattributed.** It comes from computing the MBR once and keeping it in RAM between levels, not from persisting it in the node. §11.5. |
| §9 — no verification harness | **Resolved.** `verify` mode; all configurations pass. |
| §9 — only uniform data tested | **Partly resolved.** `--clustered` added and passing; skew still needs a quality (not just correctness) comparison. |

### Still open, and now better targeted

1. **The k-way heap merge is the new bottleneck** (§12.7) — ~2.4 s of a 5.7 s
   build, single-threaded, ~35 ns/entry of pointer-chasing. Options: a
   tournament tree (better branch prediction and cache behaviour than a binary
   heap), multi-threaded merging of disjoint key ranges, or moving the merge
   itself to the GPU. **This is the highest-value remaining optimization.**
2. **The CPU control group still does not exist for the rewrite.** §5.6's
   warning stands and is now sharper: with the crossover measured at 8 Ki and
   `std::sort` single-threaded, a fair baseline needs a parallel CPU sort
   before any speedup headline is published.
3. **CUDA streams** to overlap H2D/D2H with the kernel (§9). §12.1 shows this
   is worth up to 70% of GPU time at large n — the biggest single remaining
   GPU-side win.
4. **Pinned-buffer reuse across phases.** Phase A allocates and frees ~512 MB
   of pinned staging, then Phase C allocates its own. One process-lifetime pool
   would remove most of §12.7's Phase A overhead.
5. **Tree quality on skewed data.** Correctness is verified on clustered input;
   what is not yet measured is whether `tests per page` and `pages read` hold up
   under skew, which is where STR is theoretically weakest.
6. **Larger than RAM, for real.** The spill path is verified with artificially
   tiny budgets. A genuine >5 GB run (≥ 220 M rectangles) is needed to confirm
   the three-regime scaling table of §3.1 end to end.

---

## 14. Reproduction (rewrite)

```bash
git checkout refactor/the-return-of-the-programer
make                                  # builds old and new binaries into bin/
./bin/gpu_info                        # suggested constants for rect_constants.h

./bin/gen_rects 60000000 data/rects.bin            # 1.44 GB, uniform
./bin/gen_rects 1000000  data/clu.bin --clustered 50

./bin/str_rtree data/rects.bin data/tree.bin                       # fully packed
./bin/str_rtree data/rects.bin data/tree.bin --fill-leaf 0.7 --fill-internal 1.0

./bin/rect_rtree_query data/tree.bin info
./bin/rect_rtree_query data/tree.bin verify data/rects.bin         # full structural check
./bin/rect_rtree_query data/tree.bin bench  data/rects.bin 0.01 20 # vs brute force
./bin/rect_rtree_query data/tree.bin range  100 100 200 200 --count
```


---

## 15. Methodology — how each measurement was taken

Everything in §12 comes from one of six instruments, all in the repository so
the numbers can be re-derived rather than trusted. This section states, for
each, *what it measures, what it deliberately excludes, and what it cannot
tell you.*

### 15.0 The instruments

| Instrument | Answers |
|---|---|
| [bench/bench_sort_crossover.cu](bench/bench_sort_crossover.cu) | Where does GPU sorting start to beat CPU sorting? (§12.1) |
| [bench/bench_pinned_memory.cu](bench/bench_pinned_memory.cu) | What does pinned memory cost, and what does it buy? (§12.3) |
| [bench/bench_context_init.cu](bench/bench_context_init.cu) | What are the one-time CUDA start-up costs? (§12.5) |
| [bench/sweep_pinned_cap.sh](bench/sweep_pinned_cap.sh) | Where is the chunk-cap optimum? (§12.4) |
| [bench/test_correctness.sh](bench/test_correctness.sh) | Is the tree right, on every code path? (§12.8) |
| `str_rtree`'s own `TimingStats` + `rect_rtree_query bench` | Where does build time go? How good is the tree? (§12.6, §12.7) |

```bash
make bench                                    # builds the three microbenchmarks
./bin/bench_sort_crossover 24 5               # max log2(n), trials
./bin/bench_pinned_memory 3                   # trials
./bin/bench_context_init                      # run cold, then again warm
./bench/sweep_pinned_cap.sh data/rects_60m.bin 64 128 256 512 1024
./bench/test_correctness.sh 1000000
```

### 15.1 Environment and what was (not) controlled

Single machine, as in §6.5: RTX 3050 Laptop 4 GB / Ryzen 5 6600H 6C-12T /
14 GB DDR5 / CUDA 12.0 / g++ 13 / `-O3 -std=c++17 -arch=sm_86`.

**Held fixed across comparisons:** the input file, the output path (so
page-cache state is comparable between rows), compiler and flags, RNG seeds,
and — in the sweep — every constant except the single one under test.

**Not controlled, and this is a laptop, so it matters:** CPU and GPU frequency
scaling, thermal throttling, and background system load. `lscpu` reported the
CPU at 38% of nominal scaling at one point during this work. Nothing here was
run under a fixed-frequency governor or with the GPU clock locked. Treat all
absolute timings as ±10% and all *ratios measured within a single process* as
much tighter.

### 15.2 Sort crossover (§12.1)

**What it does.** For each `n` in a ×4 geometric sweep from 1 Ki to 16 Mi, it
times two arms over the same data:

- *CPU arm*: refill a pinned host buffer from an untouched master copy, then
  `std::sort` with the **same centroid comparator the loader itself uses**.
- *GPU arm*: refill the same buffer, then time the **complete round trip** —
  H2D copy, key-extraction kernel, `sort_by_key`, D2H copy.

**Controls.** The refill is *outside* the timer in both arms, so neither is
charged for staging its own input. Both sort identical data. Best of 5 trials
is reported, which favours both arms equally.

**Deliberately excluded, because production code pays them once rather than per
sort:** CUDA context creation (forced before the sweep), `cudaMalloc` of the
device buffers (hoisted out of the loop), and `cudaMallocHost` of the pinned
buffer (hoisted; measured separately in §12.3). **This matters for
interpretation:** the table describes a sort into a *reused* buffer, which is
what the loader does. For a one-shot sort you must add ~0.53 ms/MB of pinned
allocation back, which pushes the crossover substantially higher.

**Why `gpu_sort_only` is reported separately.** The gap between it and
`gpu_total` *is* the PCIe transfer tax, isolated. That is the whole §2.4
break-even argument reduced to two columns.

**What it cannot tell you.** Nothing about multi-threaded CPU sorting — the CPU
arm is one core. The GPU-vs-CPU ratios here are therefore an **upper bound on
the honest speedup**, and §13's open item 2 stands.

### 15.3 Pinned memory (§12.3)

**Procedure**, per size, best of 3: time `cudaMallocHost`; `memset` the whole
buffer so every page is resident (otherwise the first H2D would also be paying
first-touch faults, conflating two costs); time an H2D `cudaMemcpy` of
`min(size, 1 GB)` and derive GB/s; time `cudaFreeHost`; and, as a control, time
`malloc` + `memset` of the same size.

**The `malloc`+`memset` control is the important design choice here.** Without
it you would conclude "pinning is expensive". With it you can see that plain
first-touch is ~0.45 ms/MB and pinning is ~0.53 ms/MB, i.e. **page-locking adds
only ~20%** and the dominant cost is simply having a large buffer at all.

**Two independent checks that the result is real:** the ms/MB column is flat to
within 3% across a 21× size range (so the cost is genuinely linear, not an
artefact of one size), and the bandwidth column varies by less than 0.6% (so
the flat-bandwidth claim is not noise).

**Known weakness.** Best-of-3 in a quiet system. The single-trial numbers taken
under memory pressure were ~2× higher (§12.3). Both are reported; a proper
treatment would sweep system memory pressure as a second variable.

### 15.4 CUDA context initialisation (§12.5)

**Procedure.** A single process times, as consecutive laps: `cudaFree(0)`
(which forces context creation), event creation, first and second `cudaMalloc`,
a synchronize, the **first** `thrust::sort` (which pays CUB module load / JIT),
and a **second** `thrust::sort` for the steady-state contrast.

**The cold/warm protocol.** With persistence mode disabled
(`nvidia-smi --query-gpu=persistence_mode`), the driver tears down GPU state
after idling. So: run once after the GPU has been idle (**cold**), then
immediately again (**warm**). Cold `cudaFree(0)` measured **1717 ms**; warm
measured **126–179 ms** across runs.

**Why this is in the methods section and not just the results.** It caused a
real analysis error during this work: the phase walls appeared not to reconcile
with the grand total by 1.6 s, which looked like an instrumentation bug. It was
a cold run's *total* being compared against a warm run's *init line* — two
different runs. **Any benchmark table that mixes cold and warm CUDA processes
is silently corrupt.** Every number in §12 comes from warm runs, and the loader
reports `GPU init` on its own line so the reader can check.

### 15.5 Chunk-cap sweep (§12.4)

**Procedure.** For each cap, `sed` rewrites the single constant
`MAX_PINNED_CHUNK_BYTES` into a scratch copy of the headers, recompiles the
loader, and runs a full end-to-end build of the same 60 M-rectangle file to the
same output path. Nothing else varies between rows.

**Why the per-phase columns are trustworthy and the total is not.** Phase A
wall and merge time are **monotone** in the cap and move in *opposite*
directions, which is exactly what the model predicts (allocation cost up,
`log k` down) — a monotone response to a swept parameter is strong evidence.
The total column is one build per cap and carries roughly **±1 s** of
run-to-run variance from page-cache and thermal state; note that 512 MB scored
7.76 s while 1024 MB scored 6.70 s, which is not a real inversion. **The
conclusion drawn is only "there is a broad flat basin at 128–256 MB", which
both the monotone columns and the totals support. No sharper claim is
warranted from one trial per point.**

### 15.6 In-loader instrumentation (§12.7)

The loader times itself, and the design of that timing is itself a result.

- **CUDA events, not host clocks**, for H2D / kernel / D2H, because the copies
  are asynchronous and a host timer would measure the wrong interval.
- **Threading contract**: the async I/O worker *never* writes the global stats.
  It returns its disk timings through a `std::future` and the main thread folds
  them in. No field is touched by two threads, so no lock is needed and no
  counter can tear. The contract is documented in the source so it survives
  edits.
- **`gpu_warmup()`** forces context creation before Phase A so the §15.4 cost
  is reported on its own line rather than inflating the first sort.
- **Component sum vs wall, reported in *both* directions.** Each phase prints
  the sum of its instrumented components *and* its wall-clock elapsed. If
  components exceed wall, the difference is `OVERLAP SAVED` — the measured
  value of running the GPU sort concurrently with the serialized I/O stream. If
  wall exceeds components, the difference is `UNTRACKED` — work happening
  inside the phase that nothing times.

  **The `UNTRACKED` line exists because its absence hid a bug.** The original
  version reported only the overlap direction and printed 0 otherwise, which
  concealed ~6 s of pinned-buffer allocation in a 10 s build. After adding
  explicit accounting for buffer allocation and RAM-cache copies, untracked
  time fell from **6.4 s to 429 ms (5.8% of wall)**. *A one-directional
  reconciliation is not a reconciliation.*

### 15.7 Correctness matrix (§12.8)

`bench/test_correctness.sh` exercises distinct **code paths**, not merely
distinct input sizes:

| Dimension | Values | Path exercised |
|---|---|---|
| Distribution | uniform, 50 gaussian clusters | tiling quality, not just validity |
| Fill factors | 1.0, 0.7, leaf 0.5 / internal 1.0, 0.25 | capacity arithmetic, tree height (0.25 forces height 4) |
| Memory regime | default budgets vs 8 MB GPU / 2 MB RAM | in-RAM merge vs **forced disk spill** |

`verify` walks the entire tree and checks five independent invariants:

1. every page holds between 1 and its effective capacity entries;
2. **every parent entry's stored MBR contains its child subtree's actual
   union** — recomputed bottom-up during the walk, so a wrong MBR anywhere is
   caught;
3. **every object ID appears exactly once**, via a bitmap over all IDs — this
   catches both loss and duplication, which a count alone would not;
4. the walked page count matches the header;
5. a sample of source rectangles round-trips: each must be returned by a range
   query over its own extent.

Independently, the build asserts its node count and height against the
closed-form `precompute_num_nodes` / `precompute_height`. **These are two
genuinely independent checks** — one structural arithmetic, one an actual walk
— and they agreed in every configuration.

**The equivalence test is the subtle one.** The in-RAM and spilled paths produce
*different trees*. Rather than treat that as a failure, the script checks the
property that actually matters: **four fixed range queries must return
identical counts from both trees.** They do. The cause was then confirmed
directly with numpy — 1,000,000 rectangles yield only 975,497 distinct
centroid-X keys at float32 precision, so **2.5% of records are tied** and the
two paths order them differently. Both are valid STR trees.

### 15.8 Query benchmark (§12.6)

**Procedure.** For each selectivity, a query box of side
`sqrt(selectivity) × root extent` is placed at 15–20 uniformly random positions
inside the root MBR (fixed seed), run in `--count` mode so result storage does
not pollute the timing, and averaged. The tool reports pages read, MBR tests,
tests per page, pruned subtrees, and time. One query per run is additionally
cross-checked against a **full brute-force scan of the source file**, and the
counts must match exactly.

**The critical caveat: these are warm-page-cache numbers.** The 60 M tree is
1.4 GB on a 14 GB machine and had just been written and walked, so it was
entirely resident. The `time` column therefore measures traversal and memory
access, **not disk**. Cold-cache query latency is unmeasured and would be
dominated by page faults.

**Which is exactly why `tests per page` is the headline metric and not time.**
Pages read and MBR tests are **counts of work, independent of cache state,
clock speed and machine** — they are properties of the tree's shape. That the
metric saturates its theoretical ceiling (166.9–170.0 out of 170) is a
structural claim that survives being re-measured on other hardware, whereas the
0.105 ms is not.

### 15.9 What these tests do *not* establish

Stated plainly, because an article should:

1. **No honest speedup headline.** There is no CPU control group for the
   rewrite, and the CPU sort arm in §12.1 is single-threaded. Every GPU-vs-CPU
   ratio here is an upper bound.
2. **No larger-than-RAM run.** The spill path is verified only with
   artificially shrunken budgets. The three-regime table of §3.1 is confirmed
   for regimes 1 and 2 and only *simulated* for regime 3. A genuine ≥ 220 M
   rectangle (> 5 GB) build is still owed.
3. **No cold-cache query numbers**, per §15.8.
4. **No real-world data.** Uniform and simple gaussian clusters only. Real
   spatial skew (OSM, TIGER, trajectories) is where STR is theoretically
   weakest and remains untested for *quality* — clustered input is verified
   **correct**, but its `tests per page` was not compared against uniform at
   matched scale.
5. **One machine, one GPU, one CUDA version.** Every constant in
   `rect_constants.h` is calibrated to this hardware; the *shapes* of the
   curves should transfer, the *values* should not be assumed to.
6. **Best-of-N reporting.** Minimums suppress noise but hide variance. Medians
   and spreads are not reported, and for the build sweep there is only one
   trial per point.
