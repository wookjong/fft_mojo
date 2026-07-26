#!/usr/bin/env bash
# =============================================================================
# What the mapped-address optimization saves, benchmark by benchmark.
#
#   ./scripts/map-address-bench.sh
#
# For each workload it compiles the device module three ways -- indices left as
# written, folded to base + mapped offset, and folded to the mapped address
# where the range parameter allows -- and counts the static instructions of
# each. `off` is the baseline; the last column is how much `addr` takes off it.
#
# The counts come from the asm a real launch emits (M2NDP_DUMP=asm), so the
# `addr` column uses the range parameter the host actually found rather than a
# guess. A workload still runs to completion under each; correctness is
# host-run.sh's job, not this script's.
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BENCHES="$(ls benchmarks/*.mojo | xargs -n1 basename | sed 's/\.mojo$//')"

# Instruction mnemonics are a tab then a letter; directives (.cfi, .size) are a
# tab then a dot, and labels end in a colon, so neither is counted.
count() { awk '/──── riscv ────/{f=1; next} f' | grep -cE '^	[a-z]'; }

printf "%-22s %6s %6s %6s %8s\n" workload off offset addr "saved"
tot_off=0
tot_addr=0
for b in $BENCHES; do
    off=$(M2NDP_MAP_ADDRESS=off    M2NDP_DUMP=asm ./scripts/host-run.sh "$b" 2>/dev/null | count)
    ofs=$(M2NDP_MAP_ADDRESS=offset M2NDP_DUMP=asm ./scripts/host-run.sh "$b" 2>/dev/null | count)
    adr=$(M2NDP_MAP_ADDRESS=addr   M2NDP_DUMP=asm ./scripts/host-run.sh "$b" 2>/dev/null | count)
    pct=$(awk -v o="$off" -v a="$adr" 'BEGIN{ printf (o>0)? "%.1f%%" : "-", (o-a)*100.0/o }')
    printf "%-22s %6s %6s %6s %8s\n" "$b" "$off" "$ofs" "$adr" "$pct"
    tot_off=$((tot_off + off))
    tot_addr=$((tot_addr + adr))
done
gpct=$(awk -v o="$tot_off" -v a="$tot_addr" 'BEGIN{ printf (o>0)? "%.1f%%" : "-", (o-a)*100.0/o }')
printf "%-22s %6s %6s %6s %8s\n" "TOTAL" "$tot_off" "" "$tot_addr" "$gpct"
