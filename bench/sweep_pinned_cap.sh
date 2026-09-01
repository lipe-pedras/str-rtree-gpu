#!/usr/bin/env bash
# sweep_pinned_cap.sh — end-to-end sweep of MAX_PINNED_CHUNK_BYTES.
#
# The chunk cap sets a trade-off with two opposing terms:
#   larger cap -> fewer runs (k), so a cheaper k-way merge (O(N log k))
#   larger cap -> more pinned memory, and pinning costs ~1 ms/MB
# This script rebuilds the loader at several caps and reports both terms plus
# the end-to-end wall time, so the basin can be located rather than guessed.
#
# METHOD: the ONLY thing that varies between builds is the one constant; the
# same input file and the same output path are used throughout, so page-cache
# state is comparable across rows.  One build per cap — the per-phase columns
# are monotone and trustworthy; TOTALS carry roughly +/-1 s of run-to-run
# variance on this laptop, so do not read a winner from the total column alone.
#
# Usage: bench/sweep_pinned_cap.sh <input.bin> [caps in MB...]
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT="${1:?usage: sweep_pinned_cap.sh <input.bin> [caps_MB...]}"
shift || true
CAPS=("$@"); [ ${#CAPS[@]} -eq 0 ] && CAPS=(64 128 256 512 1024)

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
# Copy every header the loader includes, not a hand-listed subset: a missing
# one makes nvcc fail and (with stderr silenced) the test silently reports an
# empty result instead of a failure.
cp src/*.h src/str_rtree_cuda.cu "$WORK/"

printf "%10s %6s %12s %12s %12s %10s\n" "cap(MB)" "runs" "phaseA(ms)" "merge(ms)" "phaseC(ms)" "total(s)"
for mb in "${CAPS[@]}"; do
  bytes=$(( mb * 1048576 ))
  sed "s/^constexpr size_t MAX_PINNED_CHUNK_BYTES = .*/constexpr size_t MAX_PINNED_CHUNK_BYTES = ${bytes}ULL;/" \
      src/rect_constants.h > "$WORK/rect_constants.h"
  if ! nvcc -O3 -std=c++17 -arch=sm_86 -I"$WORK" "$WORK/str_rtree_cuda.cu" -o "$WORK/sweep"; then
    echo "FAILED to build at cap ${mb} MB"; exit 1
  fi
  out=$("$WORK/sweep" "$INPUT" "$WORK/tree.bin" 2>/dev/null)
  printf "%10s %6s %12s %12s %12s %10s\n" "$mb" \
    "$(echo "$out" | grep 'Runs:'            | awk '{print $2}')" \
    "$(echo "$out" | grep -A9  'Phase A -'   | grep 'WALL:' | awk '{print $2}')" \
    "$(echo "$out" | grep -m1 'CPU merge:'   | awk '{print $3}')" \
    "$(echo "$out" | grep -A11 'Phase C -'   | grep 'WALL:' | awk '{print $2}')" \
    "$(echo "$out" | grep 'Total wall-clock' | awk '{print $3}')"
done
