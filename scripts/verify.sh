#!/usr/bin/env bash
# =============================================================================
# Check that the generated artifacts look the way they should.
# Run after build.sh.
#   ./scripts/verify.sh
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LLC="${LLC:-$REPO/build/llvm/bin/llc}"

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
# A µthread's index in the range is the whole of what the workloads ask about
# themselves, which is what the reference gives its kernels as well.
check "the only ID the workloads read" \
      "grep -ho '@__m2ndp_[a-z]*_*uthread_id\\|@__m2ndp_group_[a-z]*' out/*.ll | sort -u" \
      "@__m2ndp_global_uthread_id"
# The operation symbols are the other half of the contract; unlike the ID
# symbols these carry an element type, since the frontend cannot overload on
# vector type.
check "indexed vector atomic symbol" \
      "grep -c '@__m2ndp_vamoadd_i32' out/histogram.ll | tr -d ' '" "2"
# Mojo's own LLVM does not know xm2ndp and warns while dropping it from its
# subtarget, but it copies the feature string into target-features verbatim.
# That is how the marker reaches our llc, so check every module carries it.
check "xm2ndp in target-features" \
      "grep -Lc 'target-features\"=\"[^\"]*+xm2ndp' out/*.ll | wc -l | tr -d ' '" "0"
check "atomic combine, not a barrier" \
      "grep -c 'atomicrmw add' out/histogram.ll | tr -d ' '" "1"
check "relaxed ordering" \
      "grep -q 'atomicrmw add .* monotonic' out/histogram.ll && echo yes" "yes"
check "no barrier symbol" \
      "grep -c '__m2ndp_barrier' out/*.ll | grep -v ':0' | wc -l | tr -d ' '" "0"
check "RVV vsetivli" \
      "test \$(grep -c 'vsetivli' out/vector_add.s) -ge 1 && echo yes" "yes"
check "RVV vector load/add/store" \
      "test \$(grep -cE 'vle32\.v|vadd\.vv|vse32\.v' out/vector_add.s) -ge 3 && echo yes" "yes"
# An index is loaded, widened, and used to address the values -- x[col_idx[k]].
# Matched by that shape rather than by SSA numbers, which move whenever the
# kernel gains or loses a load ahead of the loop.
check "indirect access chain" \
      "grep -A1 'sext i32 .* to i64' out/spmv.ll | grep -qE 'getelementptr inbounds float, ptr %[0-9]+, i64 %[0-9]+' && echo yes" "yes"
# A kernel takes no arguments -- its parameters are in the scratchpad -- so a
# launch carries the kernel and nothing else. A second operand here would mean
# a kernel taking arguments the launcher has nowhere to put.
check "launches carry the kernel and nothing else" \
      "grep -h 'call void @__m2ndp_launch_' out/*.ll | grep -cE ', (i64|ptr )' | tr -d ' '" "0"
# The task's three kernels, internal because only device_main reaches them.
check "histogram: 3 kernels in one module" \
      "grep -cE '^define internal void @\"histogram::Histogram::(initialize|body|finalize)' out/histogram.ll | tr -d ' '" "3"
# Two: the bins the kernels share, and the task's parameters. Both are the
# task's own scratchpad, laid out by the compiler.
check "histogram: bins and params in the scratchpad" \
      "grep -c 'addrspace(3) global' out/histogram.ll | tr -d ' '" "2"
# One use per kernel, on top of the definition. Counting uses rather than
# occurrences: the body used to unroll into sixteen of them and now needs
# exactly one, so a threshold would have hidden the change either way. The
# blob is found by its size, its name being a hash.
check "histogram: all kernels hit the bins" \
      "b=\$(sed -n 's/^\\(@memory_blob_[0-9a-f]*\\) = internal addrspace(3) global \\[1024 x i8\\].*/\\1/p' out/histogram.ll); grep -c \"\$b\" out/histogram.ll | tr -d ' '" "4"
# INIT/FINAL still combine with scalar atomics; BODY is the vector one, and
# it keeps the scratchpad address space through the call.
check "histogram: scratchpad atomic" \
      "grep -q '@__m2ndp_vamoadd_i32(ptr addrspace(3)' out/histogram.ll && echo yes" "yes"
check "histogram: one vector atomic, not 16 scalar" \
      "grep -c 'atomicrmw add ptr addrspace(3)' out/histogram.ll | tr -d ' '" "0"
# The schedule lives in the workload, not the launcher: device_main launches
# init serially, the body in parallel, then final serially. Scoped to the
# exported entry point, since histogram's module also carries a mangled copy of
# it and counting the whole file would see the schedule twice.
check "device_main: 1 parallel + 2 serial launches" \
      "awk '/^define dso_local void @__m2ndp_rt_launch_task/,/^}/' out/histogram.ll | grep -c 'call void @__m2ndp_launch_' | tr -d ' '" "3"
# Conforming to NDPTask is the whole interface to the host: the task exports
# the runtime entry point and nothing else. device_main and the kernels are
# internal, which is what keeps one task per ELF from colliding with another.
check "task exports only its launch entry" \
      "grep -c '^define dso_local[^@]*@[a-z_]' out/histogram.ll | tr -d ' '" "1"
# One load and one store in the kernel, and nothing else.
# Width-agnostic: the lane count follows PACKET, and what is being checked is
# that the kernel is one load and one store, not how wide they are.
check "memcpy: one vector load, one store" \
      "test \$(grep -cE 'load <[0-9]+ x i32>' out/memcpy.ll) -eq \$(grep -cE 'store <[0-9]+ x i32>' out/memcpy.ll) && grep -cE 'load <[0-9]+ x i32>' out/memcpy.ll | tr -d ' '" "1"
check "memset: splat via shufflevector" \
      "grep -qE 'shufflevector <[0-9]+ x i8>' out/memset.ll && echo yes" "yes"
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
# Half precision is part of the modelled machine, so a conversion is one
# instruction rather than a call into compiler-rt -- which a kernel could not
# make.
check "narrow: fp32 -> fp16 in one instruction" \
      "grep -c 'vfncvt.f.f.w' out/narrow.s | tr -d ' '" "1"
check "wide: fp16 -> fp32 in one instruction" \
      "grep -c 'vfwcvt.f.f.v' out/wide.s | tr -d ' '" "1"
# Float and integer vector atomics are different instructions. This is the
# float one, which the integer symbol name would not have reached.
check "gemv: float vector atomic" \
      "grep -c 'm2ndp.vfamoaddei32.v' out/gemv_aggregation.s | tr -d ' '" "1"
# softmax reduces with both scalar float atomics, one per kernel.
check "softmax: scalar float max and add atomics" \
      "grep -cE 'm2ndp.famo(max|add).w' out/softmax.s | tr -d ' '" "2"

# Mapped-address recovery. The artifacts are the baseline -- build.sh runs llc
# with the pass off -- so the baseline reconstructs the id (a shift by the
# log2 of the 32-byte packet) and materializes the mapped array's base. Then
# the same IR compiled with the pass on drops the shift and reads the array
# straight from the mapped-address register a2. See INTERFACE.md.
M="+m,+a,+f,+d,+v,+zvl128b,+xm2ndp"
vaddr() { "$LLC" -mtriple=riscv64-unknown-elf -mattr="$M" "$@" out/vector_add.ll -o - 2>/dev/null \
    | awk '/^"vector_add::VectorAdd::body\(\)":/{f=1} f&&/^\t[a-z]/{print} f&&/^\tret$/{exit}'; }
check "map-address: baseline rebuilds the id" \
      "vaddr | grep -c 'slli.*5' | tr -d ' '" "1"
check "map-address: addr fold reads a2, no rebuild" \
      "vaddr -m2ndp-map-address=addr -m2ndp-packet=32 -m2ndp-range-param=0 \
       | grep -qE 'vle32.v\s+v[0-9]+, \(a2\)' && ! vaddr -m2ndp-map-address=addr -m2ndp-packet=32 -m2ndp-range-param=0 | grep -q 'slli' && echo yes" "yes"

echo ""
if [ "$FAIL" = 0 ]; then
    echo "[verify] all passed"
else
    echo "[verify] some checks failed"
    exit 1
fi
