# GPU R-Tree Bulk-Loading When the Dataset Does Not Fit in VRAM

An out-of-core CUDA implementation of **Sort-Tile-Recursive** R-Tree bulk loading, for
datasets larger than *both* the GPU's memory and the host's RAM, with a CPU twin used
as a control group.

Undergraduate research project (Iniciação Científica), Instituto de Engenharia de Sistemas e Tecnologia da Informação, **Universidade Federal de Itajubá** (IESTI - UNIFEI).

![Sort-Tile-Recursive bulk loading, animated](doc/media/str_animation.gif)

*1080p60 version: [`doc/media/str_animation.mp4`](doc/media/str_animation.mp4).*

> **The gap.** Every GPU R-Tree paper in [`references/`](references/) assumes the
> spatial data already lives in GPU global memory: *"we assume the spatial data can
> fit into memory, which is reasonable given that memory is getting larger and
> cheaper"* (Luo et al., 2012). Their experiments run on 10⁵–10⁶ objects. But GPU
> memory is the most expensive and slowest-growing tier in the machine, and spatial
> datasets are not shrinking. **Given that the data will not fit, what is the best
> way to use the GPU anyway?**

---

## Results

RTX 3050 Laptop (4 GB, **2.94 GB usable**), Ryzen 5 6600H (6C/12T), 16 GB RAM.

**300 M rectangles — 7.2 GB, against a 5 GB RAM budget and 2.9 GB of VRAM. 27 runs
spill to disk.**

| build | sort A | sort C | merge | **total** |
|---|---:|---:|---:|---:|
| CPU ×1 | 30 662 ms | 23 134 ms | 8 783 ms | 82.04 s |
| CPU ×6 | 13 307 ms | 8 404 ms | 10 420 ms | **50.81 s** |
| CPU ×12 | 13 476 ms | 9 304 ms | 13 755 ms | 56.65 s |
| **GPU** | **491 ms** | **715 ms** | 11 892 ms | **39.70 s** |

The index builds in **40.75 s**, producing **1 775 150 pages at height 4** (matching
the closed form exactly) and verifying **300 000 000 / 300 000 000 objects indexed
exactly once at 100.00% leaf occupancy**.

Range queries on that tree extract **169.5–170.0 pruning decisions per page fetched,
against a ceiling of 170**. At 0.01% selectivity: same results as a brute-force scan,
**8 386× faster** (0.40 ms vs 3 393.67 ms).

---

## Findings

### 1. The workload scores terribly on the roofline model and wins anyway

| Operation | Work per byte | vs. ridge point (32 FLOP/B) |
|---|---|---|
| Merge-sort pass | ≈ 0.06 op/byte | ~500× below |
| Leaf MBR reduction | 0.5 op/byte | ~64× below |
| Range-query MBR test | 0.125 op/byte | ~250× below |

Spatial indexing has no reuse to exploit: each rectangle is read, compared once or
twice, written. By the standard metric this is close to the worst possible GPU
candidate. It wins anyway because the real advantage is bandwidth, not FLOPs — 192 GB/s
of GDDR6 against 76.8 GB/s of theoretical DDR5, and a regular memory-bound kernel
reaches >80% of its peak where latency-bound CPU cores reach a fraction of a lower one.

So the figure of merit is not FLOP/byte but **effective bytes per second**, and the
engineering problem is never "add arithmetic" but **move each byte across the fastest
tier that can hold it, and touch it the fewest possible times.**

### 2. Amdahl's law

**GPU sorting is 18.0× faster than all six CPU cores. The end-to-end build is 1.28×
faster.** Sorting is 42.7% of the CPU build and 3.0% of the GPU build; the rest is data
movement (disk read 23.3 s, merge 11.7 s). The sorting fraction *shrinks* as data grows,
so the ratio gets worse with scale.

This is why [`src/str_rtree_cpu.cpp`](src/str_rtree_cpu.cpp) exists: identical phases,
on-disk format and merge code, **only the sort differs**. Without it there is no honest
denominator.

### 3. The GPU is a net loss below ~8 M rectangles

| N | GPU | CPU ×1 | CPU ×6 | verdict |
|---:|---:|---:|---:|---|
| 1 M | 0.31 s | 0.21 s | 0.14 s | GPU loses ~2× |
| 8 M | 0.94 s | 1.75 s | 0.91 s | **tie with all-core CPU** |
| 16 M | 1.64 s | 3.87 s | 1.83 s | GPU wins |

The raw sort crossover is at ~8 Ki entries; a whole *build* breaks even only at ~8 M
rectangles (190 MB). The gap is fixed cost: ~126 ms of warm CUDA context creation
(1.7 s cold), pinned allocation at ~0.53 ms/MB, and crossing the bus twice at a measured
13.0–13.4 GB/s — 84% of PCIe 4.0 x8 peak (this laptop wires the board at x8).

### 4. The merge got 2.8× from a better algorithm and 3.4× from an allocator

Once the sort runs on the GPU, the k-way merge is the largest cost in the build, so it
got its own benchmark ([`bench/bench_kway_merge.cu`](bench/bench_kway_merge.cu)). Two
separate things came out of it.

**The algorithm.** A binary heap does one comparison per level per element and chases
pointers doing it. A loser tree halves the comparisons and branches predictably; the
runs can then be split at binary-searched key boundaries so threads write to disjoint
output ranges, with no locks and no merging of partial results. On the real 60 M build:

| Phase B algorithm | merge |
|---|---:|
| binary heap | 2365 ms |
| loser tree, 1 thread | 1469 ms |
| partitioned loser trees, 12 threads | **834 ms** |

**The allocation.** With the merge at 834 ms, the remaining cost turned out not to be
in the merge at all, but in preparing the buffer it writes into. `std::vector::resize()`
value-initialises: before a single element is merged, it writes zeros across the whole
1.4 GB output and faults in every page, on one thread.

| N = 64 M, k = 6 — testbench | ms |
|---|---:|
| merge into a buffer already allocated and faulted in | **182.7** |
| `resize()` a fresh output vector, before merging | **639.0** |
| total time | **843.1** |

**Preparing the buffer cost 3.5× the merge that used it.** The fix is an allocator that
leaves trivially-constructible elements uninitialised — `Entry` is trivially
constructible and every element is written before it is read, so nothing is lost. The
zero-fill disappears and the pages get faulted in by the 12 merge threads that were
going to overwrite them anyway. Phase B drops **834 ms → 242 ms**.

Together: **Phase B 9.8× faster, the whole 60 M build 5.74 s → 3.25 s**, and the larger
half of that came from the allocator rather than the algorithm.

### 5. Merging does not belong on the GPU

The argument that justifies GPU sorting rules out GPU merging: sorting does O(log n)
passes and clears the bandwidth bar, a merge is one pointer-chasing pass. Measured, the
GPU merge is **2.3× slower** than the CPU's partitioned merge.

<details>
<summary><b>Other findings</b></summary>

- **`thrust::sort` was silently not using radix sort.** Sorting `Entry` by a comparator
  dispatches to merge sort; extracting a `uint32` key and calling `sort_by_key` reaches
  `DeviceRadixSort`. 16.7 M entries in 26 ms, ~4×.
- **The centroid comparator uses `min + max`, never `(min + max)/2`.** Division is
  monotone: pure waste in a comparator called O(n log n) times.
- **Keeping each node's MBR in RAM between levels is what saves a pass** — not
  persisting it in the node, which costs a permanent query-side tax (pruning ceiling 128
  instead of 170). `Entry` is both the sort element and the packed-node entry, so each
  level's output array *is* the next level's sort input and the raw data is read exactly
  once however tall the tree gets.
- **Chunk cap is a two-sided trade**, flat optimum at 128–256 MB: Phase A rises with the
  cap (pinned allocation), the merge falls (fewer runs, lower `log k`). 256 MB cut the
  60 M build by 32%.
- **More I/O threads made it slower.** Two concurrent read streams degrade the SSD. A
  single serialized I/O thread double-buffered against the GPU saved 7 345 ms on the
  300 M build (48 090 ms of component cost in a 40 746 ms wall).
- **SMT does not help.** CPU ×12 is no faster than ×6, and at 300 M it is slower because
  the merge contends.

</details>

---

## How it works

Four phases over one abstraction: a **cached run** that is either RAM-resident or
disk-resident, chosen greedily against a byte budget, so a single code path spans "fits
in VRAM" → "fits in RAM" → "fits nowhere".

| Phase | What it does | Where |
|---|---|---|
| **A** | Sort chunks by centroid X into sorted runs | GPU + I/O overlap |
| **B** | Single-pass k-way merge of the runs | CPU, loser trees |
| **C** | Sort each vertical slice by centroid Y; pack leaves | GPU + I/O overlap |
| **D** | Re-run STR over the node level; build up to the root | CPU, in RAM |

**On-disk format.** Page 0 is the header; every node is one aligned 4096-byte page of
`{ PageHeader, Entry[] }`, an `Entry` being `(child MBR, child id)` at 24 bytes, so
`MAX_ENTRIES_PER_PAGE = (4096 − 8) / 24 = 170`. A node stores **its children's** MBRs,
so one page fetch yields every pruning decision for that node's fan. The root's MBR,
having no parent to live in, sits in the file header beside an explicit `root_page`.

---

## Building and running

Requires CUDA (`nvcc`), a C++17 compiler, and an NVIDIA GPU. `NVCCFLAGS` targets
`sm_86`; adjust in the [`Makefile`](Makefile) for other architectures.

```bash
make                                   # all binaries -> bin/
./bin/gpu_info                         # suggested constants for src/rect_constants.h

# Generate data
./bin/gen_rects 60000000 data/rects.bin              # 1.44 GB, uniform
./bin/gen_rects 1000000  data/clu.bin --clustered 50 # 50 gaussian clusters

# Build the index
./bin/str_rtree     data/rects.bin data/tree.bin
./bin/str_rtree     data/rects.bin data/tree.bin --fill-leaf 0.7 --fill-internal 1.0
./bin/str_rtree_cpu data/rects.bin data/tree.bin --threads 6    # control group

# Query it
./bin/rect_rtree_query data/tree.bin info
./bin/rect_rtree_query data/tree.bin verify data/rects.bin      # structural check
./bin/rect_rtree_query data/tree.bin bench  data/rects.bin 0.01 20
./bin/rect_rtree_query data/tree.bin range  100 100 200 200 --count
```

`make help` lists every target and flag. `make render` rebuilds the animation.

### Benchmarks and correctness

Every number above comes from an instrument in the repository, so it can be re-derived
rather than trusted.

```bash
make bench
./bin/bench_sort_crossover 24 5          # GPU vs CPU sort break-even
./bin/bench_pinned_memory 3              # pinned alloc cost vs H2D bandwidth
./bin/bench_context_init                 # CUDA start-up, run cold then warm
./bin/bench_kway_merge --n 16 64 --k 4 16 --disk
./bench/sweep_pinned_cap.sh  data/rects.bin
./bench/compare_gpu_cpu.sh   data/rects.bin
./bench/test_correctness.sh              # the full matrix
```

`verify` walks the whole tree: every page holds 1..cap entries, every parent entry's
MBR contains its child subtree's actual union, every object ID appears exactly once,
the walked page count matches the header, and sampled rectangles round-trip through a
range query over their own extent. All configurations pass — four fill factors,
clustered data, the forced-spill path, in-RAM vs spilled equivalence, and the 300 M
build.

The in-RAM and spilled paths produce **different trees but identical query answers**:
at float32 precision 2.5% of records share an exact centroid key, so the two paths
break ties differently. Both are valid STR trees.

---

## Repository layout

| Path | Contents |
|---|---|
| [`src/`](src/) | `str_rtree_cuda.cu` (GPU loader), `str_rtree_cpu.cpp` (CPU twin), `rect_rtree_query.cpp`, `kway_merge.h`, `rect_rtree_format.h`, generators, `gpu_info.cu` |
| [`bench/`](bench/) | Microbenchmarks, correctness and comparison scripts |
| [`animation/`](animation/) | Manim scene, Python STR reference, tree-binary reader |
| [`doc/`](doc/) | `resumo_expandido.md` (extended abstract, PT-BR) and rendered media |
| [`references/`](references/) | The papers the gap was found in |

---

## References

- **GUTTMAN, A.** R-trees: a dynamic index structure for spatial searching. *ACM SIGMOD*, 1984, p. 47–57.
- **LEUTENEGGER, S. T.; EDGINGTON, J. M.; LOPEZ, M. A.** STR: a simple and efficient algorithm for R-tree packing. ICASE Report No. 97-14 / NASA CR-201661, 1997.
- **LUO, L.; WONG, M. D. F.; LEONG, L.** Parallel implementation of R-trees on the GPU. *ASP-DAC*, 2012, p. 353–358.
- **SHAHI, N.** Optimization of R-tree construction for spatial data using GPU. M.Sc. dissertation, Southern Illinois University Edwardsville, 2025.
- **WILLIAMS, S.; WATERMAN, A.; PATTERSON, D.** Roofline: an insightful visual performance model for multicore architectures. *CACM*, v. 52, n. 4, p. 65–76, 2009.

---

*The extended abstract for the IX Simpósio de IC is at
[`doc/resumo_expandido.md`](doc/resumo_expandido.md).*
