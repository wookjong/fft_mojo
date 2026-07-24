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
fi

# This is a smoke test, not the functional one. It asks whether the pipeline
# and the extension stand up, from hand-written assembly. Whether a compiled
# workload gets the right answer is scripts/host-run.sh: a host program names a
# task, and it is compiled, run and checked from there.

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "[smoke] some checks failed"
    exit 1
fi
echo "[smoke] passed"
