#!/usr/bin/env bash
# =============================================================================
# Check that the generated artifacts look the way they should.
# Run after build.sh.
#   ./scripts/verify.sh
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FAIL=0
check() {
    local desc="$1" cmd="$2" expect="$3"
    local got
    got="$(eval "$cmd" 2>/dev/null)"
    if [ "$got" = "$expect" ]; then
        printf "  OK   %-44s (%s)\n" "$desc" "$got"
    else
        printf "  FAIL %-44s expected=%s got=%s\n" "$desc" "$expect" "$got"
        FAIL=1
    fi
}

echo "[verify] checking artifacts"
[ -d out ] || { echo "no out/. Run ./scripts/build.sh first"; exit 1; }

check "RISC-V target" \
      "grep -h 'target triple' out/*.ll | sort -u | grep -c riscv64" "1"
check "all 4 M2NDP ID symbols" \
      "grep -ho '@__m2ndp_[a-z]*_*uthread_id\\|@__m2ndp_group_[a-z]*' out/*.ll | sort -u | wc -l | tr -d ' '" "4"
# The operation symbols are the other half of the contract; unlike the ID
# symbols these carry an element type, since the frontend cannot overload on
# vector type.
check "indexed vector atomic symbol" \
      "grep -c '@__m2ndp_vamoadd_i32' out/histogram.ll | tr -d ' '" "2"
# Mojo's own LLVM does not know xm2ndp and warns while dropping it from its
# subtarget, but it copies the feature string into target-features verbatim.
# That is how the marker reaches our llc, so check every module carries it.
check "xm2ndp in target-features" \
      "grep -lc 'target-features\"=\"[^\"]*+xm2ndp' out/*.ll | wc -l | tr -d ' '" "6"
check "atomic combine, not a barrier" \
      "grep -c 'atomicrmw fadd' out/spmv.ll | tr -d ' '" "1"
check "relaxed ordering" \
      "grep -q 'atomicrmw fadd .* monotonic' out/spmv.ll && echo yes" "yes"
check "no barrier symbol" \
      "grep -c '__m2ndp_barrier' out/*.ll | grep -v ':0' | wc -l | tr -d ' '" "0"
check "RVV vsetivli" \
      "test \$(grep -c 'vsetivli' out/vector_add.s) -ge 1 && echo yes" "yes"
check "RVV vector load/add/store" \
      "test \$(grep -cE 'vle32\.v|vadd\.vv|vse32\.v' out/vector_add.s) -ge 3 && echo yes" "yes"
check "indirect access chain" \
      "grep -q 'getelementptr inbounds float, ptr %2, i64' out/spmv.ll && echo yes" "yes"
check "histogram: 3 phases in one module" \
      "grep -c '^define dso_local void @histogram_' out/histogram.ll | tr -d ' '" "3"
check "histogram: one shared scratchpad global" \
      "grep -c 'addrspace(3) global' out/histogram.ll | tr -d ' '" "1"
# One use per phase, on top of the definition. Counting uses rather than
# occurrences: the body used to unroll into sixteen of them and now needs
# exactly one, so a threshold would have hidden the change either way.
check "histogram: all phases hit that global" \
      "grep -c 'memory_blob' out/histogram.ll | tr -d ' '" "4"
# INIT/FINAL still combine with scalar atomics; BODY is the vector one, and
# it keeps the scratchpad address space through the call.
check "histogram: scratchpad atomic" \
      "grep -q '@__m2ndp_vamoadd_i32(ptr addrspace(3)' out/histogram.ll && echo yes" "yes"
check "histogram: one vector atomic, not 16 scalar" \
      "grep -c 'atomicrmw add ptr addrspace(3)' out/histogram.ll | tr -d ' '" "0"
check "memcpy: one vector load + store" \
      "test \$(grep -cE 'load <8 x i32>|store <8 x i32>' out/memcpy.ll) -eq 2 && echo yes" "yes"
check "memset: splat via shufflevector" \
      "grep -q 'shufflevector <32 x i8>' out/memset.ll && echo yes" "yes"
check "imdb: predicate selects vmslt.vx" \
      "grep -q 'vmslt.vx' out/imdb_lt_int64.s && echo yes" "yes"

echo ""
if [ "$FAIL" = 0 ]; then
    echo "[verify] all passed"
else
    echo "[verify] some checks failed"
    exit 1
fi
