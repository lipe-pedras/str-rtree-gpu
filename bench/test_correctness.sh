#!/usr/bin/env bash
# test_correctness.sh — the correctness matrix behind the VERIFY OK claims.
#
# Exercises every distinct CODE PATH, not just every input size:
#   * uniform vs clustered data                (tiling quality, not just validity)
#   * four fill factors                        (capacity arithmetic, tree height)
#   * in-RAM path vs forced disk-spill path    (cached runs, k-way merge, spill)
#   * in-RAM vs spill EQUIVALENCE               (same answers from different trees)
#
# `verify` walks the entire tree and checks:
#   - every page holds 1..cap entries
#   - every parent entry's stored MBR contains its child subtree's actual union
#   - every object id appears EXACTLY ONCE (bitmap over all ids)
#   - the walked page count matches the header
#   - a sample of source rectangles round-trips through a range query
# and the build itself asserts node count and height against the closed-form
# precompute_num_nodes / precompute_height.
#
# Usage: bench/test_correctness.sh [N]
set -uo pipefail
cd "$(dirname "$0")/.."
N="${1:-1000000}"
fail=0
note() { printf "%-52s %s\n" "$1" "$2"; [ "$2" = "VERIFY OK" ] || fail=1; }

mkdir -p data tmp
./bin/gen_rects "$N" data/t_uni.bin                >/dev/null 2>&1
./bin/gen_rects "$N" data/t_clu.bin --clustered 50 >/dev/null 2>&1

for f in "--fill 1.0" "--fill 0.7" "--fill-leaf 0.5 --fill-internal 1.0" "--fill 0.25"; do
  ./bin/str_rtree data/t_uni.bin data/t_tree.bin $f >/dev/null 2>&1
  note "uniform  $f" "$(./bin/rect_rtree_query data/t_tree.bin verify data/t_uni.bin 2>/dev/null | tail -1)"
done

./bin/str_rtree data/t_clu.bin data/t_clu_tree.bin >/dev/null 2>&1
note "clustered (50 gaussian clusters)" \
     "$(./bin/rect_rtree_query data/t_clu_tree.bin verify data/t_clu.bin 2>/dev/null | tail -1)"

# Forced external path: shrink the budgets so runs must spill to disk.
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
# Copy every header the loader includes, not a hand-listed subset: a missing
# one makes nvcc fail and (with stderr silenced) the test silently reports an
# empty result instead of a failure.
cp src/*.h src/str_rtree_cuda.cu "$WORK/"
sed -e 's/^constexpr size_t USABLE_GPU_BYTES = .*/constexpr size_t USABLE_GPU_BYTES = 8388608ULL;/' \
    -e 's/^constexpr size_t USABLE_RAM_BYTES = .*/constexpr size_t USABLE_RAM_BYTES = 2097152ULL;/' \
    -e 's/^constexpr size_t MAX_PINNED_CHUNK_BYTES = .*/constexpr size_t MAX_PINNED_CHUNK_BYTES = 4194304ULL;/' \
    -e 's/^constexpr size_t PHASE_LEAF_HOST_BYTES = .*/constexpr size_t PHASE_LEAF_HOST_BYTES = 4194304ULL;/' \
    src/rect_constants.h > "$WORK/rect_constants.h"
if ! nvcc -O3 -std=c++17 -arch=sm_86 -I"$WORK" "$WORK/str_rtree_cuda.cu" -o "$WORK/tiny"; then
  echo "FAILED to build the reduced-budget loader; cannot test the spill path"; exit 1
fi
./bin/str_rtree data/t_uni.bin data/t_ram.bin >/dev/null 2>&1
"$WORK/tiny" data/t_uni.bin data/t_ext.bin >/dev/null 2>&1
note "forced disk spill (8 MB GPU / 2 MB RAM budget)" \
     "$(./bin/rect_rtree_query data/t_ext.bin verify data/t_uni.bin 2>/dev/null | tail -1)"

# The two paths break ties differently, so the TREES differ; the ANSWERS must not.
echo
echo "in-RAM vs spilled path — query equivalence:"
for box in "100 100 200 200" "0 0 1010 1010" "500 500 505 505" "12.5 900.25 33.75 950.5"; do
  a=$(./bin/rect_rtree_query data/t_ram.bin range $box --count 2>/dev/null | grep 'Found:' | tr -dc 0-9)
  b=$(./bin/rect_rtree_query data/t_ext.bin range $box --count 2>/dev/null | grep 'Found:' | tr -dc 0-9)
  s=MATCH; [ "$a" = "$b" ] || { s=MISMATCH; fail=1; }
  printf "  [%-24s] in-RAM=%-9s spilled=%-9s %s\n" "$box" "$a" "$b" "$s"
done

echo
[ $fail -eq 0 ] && echo "ALL CORRECTNESS TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
