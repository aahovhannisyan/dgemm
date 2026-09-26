#!/usr/bin/env bash
# Correctness sweep for bin/summa: grid shapes, uneven blocks, panel widths
# that straddle block boundaries, tiny N, and blocking broadcasts.
set -u
cd "$(dirname "$0")/.."

MPIRUN=${MPIRUN:-mpirun}
MPIFLAGS=${MPIFLAGS:---oversubscribe}
BIN=./bin/summa

# ranks  grid  N     b    extra
cases=(
  "1  -    1000  512  "
  "2  -    1031  100  "
  "3  -    1000  64   "
  "4  -    2050  512  "
  "4  1x4  1031  128  "
  "4  4x1  1031  128  "
  "6  -    1000  77   "
  "6  2x3  1000  512  "
  "8  2x4  1537  200  "
  "6  2x3  3     512  "
  "6  3x2  7     2    "
  "4  -    1000  64   --no-overlap"
  "6  2x3  1031  100  --no-overlap"
)

pass=0; fail=0
for c in "${cases[@]}"; do
  read -r np grid n b extra <<<"$c"
  args=(-n "$n" -b "$b" -t 2 -r 1 -w 0 --verify $extra)
  [[ $grid != - ]] && args+=(-g "$grid")
  if out=$($MPIRUN $MPIFLAGS -np "$np" "$BIN" "${args[@]}" 2>&1) && grep -q "(OK)" <<<"$out"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "FAIL: np=$np ${args[*]}"
    echo "$out" | sed 's/^/    /'
  fi
done

echo "$pass/$((pass + fail)) cases passed"
[[ $fail -eq 0 ]]
