#!/usr/bin/env bash
# =============================================================================
# Run the Spike smoke test: assemble, link, execute, check the exit code.
#
#   ./scripts/spike-smoke.sh
#
# Nothing M²NDP-specific is exercised. The point is to have the pipeline stand
# up on its own, so that a failure after the extension lands is the extension.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SPIKE="${SPIKE:-$REPO/build/spike/install/bin/spike}"
LLC="${LLC:-$REPO/build/llvm/bin/llc}"
EXTLIB="${EXTLIB:-$REPO/build/spike/libm2ndp_ext.so}"
# libriscv.so is not on the default search path.
export LD_LIBRARY_PATH="$REPO/build/spike/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LLVM_MC="${LLVM_MC:-$REPO/build/llvm/bin/llvm-mc}"
# lld, not a distribution binutils: the linker has to understand the ISA
# string this LLVM emits, and ours rejects `zmmul` the moment two toolchains'
# objects are merged.
LD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"

# Matches m2ndp_target() in src/m2ndp.mojo, minus the vendor extension, which
# Spike does not know about yet.
FEATURES="+m,+a,+f,+d,+c,+v,+zvl128b"
ISA="rv64gcv_zvl128b"
# scripts/m2ndp.lds puts code at 0x10000, which is below where Spike puts
# memory by default, so it has to be told. The region stops short of
# 0x2000000, where Spike's CLINT lives -- overlapping it is a startup error.
MEM="-m0x10000:0x1ff0000"
# The generated tests reach f16, which the benchmarks do not.
FEATURES_FP16="$FEATURES,+zfh"
ISA_FP16="${ISA}_zfh"

fail() { echo "  FAIL $*"; exit 1; }

for t in "$SPIKE" "$LLVM_MC"; do
    [ -x "$t" ] || fail "$t not found. Run ./scripts/build-spike.sh and ./scripts/build-llvm.sh"
done
[ -x "$LD" ] || fail "$LD not found. Run ./scripts/build-llvm.sh"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Both --extlib and --extension are needed: the first loads the library, the
# second turns the extension on. With only the first, the instructions are
# still illegal.
run_test() {
    local name="$1" src="$2" feat="$3"; shift 3
    local TEST_ISA="${TEST_ISA:-$ISA}"
    printf "  %-28s " "$name"
    if ! "$LLVM_MC" -triple=riscv64 -mattr="$feat" -filetype=obj \
            "$src" -o "$OUT/t.o" 2>"$OUT/err"; then
        echo "FAIL (assembly)"; sed 's/^/    /' "$OUT/err"; return 1
    fi
    # The task's own link script, not a simulator-specific one: the layout
    # under test should be the layout the compiler was built against.
    if ! "$LD" -T scripts/m2ndp.lds -e _start "$OUT/t.o" -o "$OUT/t.elf" \
            2>"$OUT/err"; then
        echo "FAIL (link)"; sed 's/^/    /' "$OUT/err"; return 1
    fi
    timeout 120 "$SPIKE" $MEM "$@" --isa="$TEST_ISA" "$OUT/t.elf" > "$OUT/log" 2>&1
    local rc=$?
    if [ "$rc" -eq 124 ]; then
        echo "FAIL (timed out -- tohost was never written)"; return 1
    elif [ "$rc" -ne 0 ]; then
        echo "FAIL (target reported $rc)"; sed 's/^/    /' "$OUT/log"; return 1
    fi
    echo "OK"
}

echo "[smoke] checking"
FAILED=0
run_test "RVV through the pipeline" sim/smoke.s "$FEATURES" || FAILED=1

if [ -f "$EXTLIB" ]; then
    run_test "indexed vector atomic" sim/smoke-vamo.s "$FEATURES,+xm2ndp" \
        --extlib="$EXTLIB" --extension=m2ndp || FAILED=1

    # Every instruction, against values computed in Python rather than
    # restated by hand. The exit code is the first test that disagreed, so a
    # failure names the instruction instead of just saying something is wrong.
    python3 sim/gen-tests.py > "$OUT/generated.s" || fail "generating tests"
    TEST_ISA="$ISA_FP16" run_test "all 64 instructions" "$OUT/generated.s" \
        "$FEATURES_FP16,+xm2ndp" --extlib="$EXTLIB" --extension=m2ndp || FAILED=1


    # Compiled kernels, not hand-written assembly: the launcher half of the
    # contract -- scratchpad region, base pointer, identity registers -- is in
    # sim/gen-kernel-test.py.
    for bench in vector_add histogram; do
        if [ ! -f "out/$bench.ll" ]; then
            echo "  $bench                     SKIP (run ./scripts/build.sh)"
            continue
        fi
        printf "  %-28s " "$bench (compiled)"
        if ! "$LLC" -mtriple=riscv64-unknown-elf \
                -mattr=+m,+a,+f,+d,+v,+zvl128b,+xm2ndp -filetype=obj \
                "out/$bench.ll" -o "$OUT/k.o" 2>"$OUT/err"; then
            echo "FAIL (llc)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
        fi
        if ! python3 sim/gen-kernel-test.py "$bench" > "$OUT/l.s" 2>"$OUT/err"; then
            echo "FAIL (generating the launcher)"; sed 's/^/    /' "$OUT/err"
            FAILED=1; continue
        fi
        if ! "$LLVM_MC" -triple=riscv64 -mattr="$FEATURES,+xm2ndp" \
                -filetype=obj "$OUT/l.s" -o "$OUT/l.o" 2>"$OUT/err"; then
            echo "FAIL (assembling the launcher)"; sed 's/^/    /' "$OUT/err"
            FAILED=1; continue
        fi
        if ! "$LD" -T scripts/m2ndp.lds -e _start "$OUT/l.o" "$OUT/k.o" \
                -o "$OUT/k.elf" 2>"$OUT/err"; then
            echo "FAIL (link)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
        fi
        timeout 120 "$SPIKE" $MEM --extlib="$EXTLIB" --extension=m2ndp \
            --isa="$ISA" "$OUT/k.elf" > "$OUT/log" 2>&1
        rc=$?
        if [ "$rc" -eq 124 ]; then
            echo "FAIL (timed out)"; FAILED=1
        elif [ "$rc" -ne 0 ]; then
            echo "FAIL (wrong result, target reported $rc)"
            sed 's/^/    /' "$OUT/log"; FAILED=1
        else
            echo "OK"
        fi
    done
fi

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "[smoke] some checks failed"
    exit 1
fi
echo "[smoke] passed"
