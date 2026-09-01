#!/usr/bin/env bash
# compare_gpu_cpu.sh — the control-group comparison.
#
# str_rtree_cpu is a faithful twin of str_rtree_cuda: identical phases, format,
# cached-run/merge/double-buffer machinery, and (since the merge was factored
# into kway_merge.h) identical merge code.  ONLY the sort differs.  So the
# difference between rows isolates the sort and the PCIe transfers it costs.
#
# Reports the two sort phases separately from the merge and the total, because
# the headline moves a long way depending on which you quote:
#   * GPU vs single-threaded CPU flatters the GPU and is the number to distrust
#   * GPU vs all-core CPU is the honest sort comparison
#   * end-to-end totals show how little of either survives Amdahl's law
#
# Usage: bench/compare_gpu_cpu.sh <input.bin> [more inputs...]
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: compare_gpu_cpu.sh <input.bin> [...]"; exit 1; }

NPROC=$(nproc)
printf "%-22s %-22s %10s %10s %10s %10s %9s\n" \
       "dataset" "build" "sortA(ms)" "sortC(ms)" "merge(ms)" "total(s)" "vs CPU1"

for input in "$@"; do
  ds=$(basename "$input")
  base=""
  for cfg in "cpu 1" "cpu $((NPROC/2))" "cpu $NPROC" "gpu 0"; do
    set -- $cfg
    if [ "$1" = "cpu" ]; then
      out=$(./bin/str_rtree_cpu "$input" data/_cmp.bin --threads "$2" 2>/dev/null)
      label="CPU --threads $2"
      sa=$(echo "$out" | grep -A6 'Phase A' | grep 'CPU sort:' | awk '{print $3}')
      sc=$(echo "$out" | grep -A6 'Phase C' | grep 'CPU sort:' | awk '{print $3}')
    else
      out=$(./bin/str_rtree "$input" data/_cmp.bin 2>/dev/null)
      label="GPU"
      sa=$(echo "$out" | grep -A8 'Phase A' | grep 'GPU sort:' | awk '{print $3}')
      sc=$(echo "$out" | grep -A8 'Phase C' | grep 'GPU sort:' | awk '{print $3}')
    fi
    mg=$(echo "$out" | grep -m1 'CPU merge:' | awk '{print $3}'); [ -z "$mg" ] && mg="-"
    tt=$(echo "$out" | grep 'Total wall' | awk '{print $3}')
    [ -z "$base" ] && base=$tt
    sp=$(python3 -c "print(f'{$base/$tt:.2f}x')" 2>/dev/null || echo "-")
    printf "%-22s %-22s %10s %10s %10s %10s %9s\n" "$ds" "$label" "$sa" "$sc" "$mg" "$tt" "$sp"
  done
  echo
done
rm -f data/_cmp.bin
