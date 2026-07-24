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
# The task's three kernels. None is exported: a kernel named as a value becomes
# a closure, and that closure is what device_main launches, so the exported
# original would only be a second unused copy of the same code.
check "histogram: 3 kernels in one module" \
      "grep -c '^define internal void @\"histogram::Histogram::device_main.*_closure_' out/histogram.ll | tr -d ' '" "3"
check "histogram: one shared scratchpad global" \
      "grep -c 'addrspace(3) global' out/histogram.ll | tr -d ' '" "1"
# One use per kernel, on top of the definition. Counting uses rather than
# occurrences: the body used to unroll into sixteen of them and now needs
# exactly one, so a threshold would have hidden the change either way.
check "histogram: all kernels hit that global" \
      "grep -c 'memory_blob' out/histogram.ll | tr -d ' '" "4"
# INIT/FINAL still combine with scalar atomics; BODY is the vector one, and
# it keeps the scratchpad address space through the call.
check "histogram: scratchpad atomic" \
      "grep -q '@__m2ndp_vamoadd_i32(ptr addrspace(3)' out/histogram.ll && echo yes" "yes"
check "histogram: one vector atomic, not 16 scalar" \
      "grep -c 'atomicrmw add ptr addrspace(3)' out/histogram.ll | tr -d ' '" "0"
# The schedule lives in the workload, not the launcher: device_main launches
# init serially, the body in parallel, then final serially.
check "device_main: 1 parallel + 2 serial launches" \
      "grep -c 'call void @__m2ndp_launch_' out/histogram.ll | tr -d ' '" "3"
# Conforming to NDPTask is the whole interface to the host: the task exports
# the runtime entry point and nothing else. device_main and the kernels are
# internal, which is what keeps one task per ELF from colliding with another.
check "task exports only its launch entry" \
      "grep -c '^define dso_local[^@]*@[a-z_]' out/histogram.ll | tr -d ' '" "1"
# One load and one store in the kernel, and nothing else.
check "memcpy: one vector load, one store" \
      "test \$(grep -c 'load <8 x i32>' out/memcpy.ll) -eq \$(grep -c 'store <8 x i32>' out/memcpy.ll) && grep -c 'load <8 x i32>' out/memcpy.ll | tr -d ' '" "1"
check "memset: splat via shufflevector" \
      "grep -q 'shufflevector <32 x i8>' out/memset.ll && echo yes" "yes"
# The assembly is now our llc's, so it shows the M2NDP lowering rather than
# what a stock LLVM would have made of the same IR. These three could not be
# checked at all while out/*.s came from the frontend's own backend.
check "IDs are register reads, not calls" \
      "grep -c 'call.*__m2ndp_\(local\|global\|group\)' out/*.s | grep -v ':0' | wc -l | tr -d ' '" "0"
# The kernel keeps no frame. Scoped to the launched copy on purpose: the
# runtime entry point above it is controller code and has one, because it
# calls.
check "the kernel keeps no frame" \
      "awk '/_closure_0/,/Lfunc_end0/' out/memcpy.s | grep -c 'addi.*sp, sp, -' | tr -d ' '" "0"
check "histogram: the vector atomic is one instruction" \
      "grep -c 'm2ndp.vamoaddei32.v' out/histogram.s | tr -d ' '" "1"
check "imdb: predicate selects vmslt.vx" \
      "grep -q 'vmslt.vx' out/imdb_lt_int64.s && echo yes" "yes"

echo ""
if [ "$FAIL" = 0 ]; then
    echo "[verify] all passed"
else
    echo "[verify] some checks failed"
    exit 1
fi
