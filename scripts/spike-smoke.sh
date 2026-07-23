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
LD="${RISCV_LD:-riscv64-unknown-elf-ld}"

# Matches m2ndp_target() in src/m2ndp.mojo, minus the vendor extension, which
# Spike does not know about yet.
FEATURES="+m,+a,+f,+d,+c,+v,+zvl128b"
ISA="rv64gcv_zvl128b"

fail() { echo "  FAIL $*"; exit 1; }

for t in "$SPIKE" "$LLVM_MC"; do
    [ -x "$t" ] || fail "$t not found. Run ./scripts/build-spike.sh and ./scripts/build-llvm.sh"
done
command -v "$LD" >/dev/null || \
    fail "$LD not found. apt-get install -y binutils-riscv64-unknown-elf"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Both --extlib and --extension are needed: the first loads the library, the
# second turns the extension on. With only the first, the instructions are
# still illegal.
run_test() {
    local name="$1" src="$2" feat="$3"; shift 3
    printf "  %-28s " "$name"
    if ! "$LLVM_MC" -triple=riscv64 -mattr="$feat" -filetype=obj \
            "$src" -o "$OUT/t.o" 2>"$OUT/err"; then
        echo "FAIL (assembly)"; sed 's/^/    /' "$OUT/err"; return 1
    fi
    if ! "$LD" -T sim/m2ndp.ld "$OUT/t.o" -o "$OUT/t.elf" 2>"$OUT/err"; then
        echo "FAIL (link)"; sed 's/^/    /' "$OUT/err"; return 1
    fi
    timeout 60 "$SPIKE" "$@" --isa="$ISA" "$OUT/t.elf" > "$OUT/log" 2>&1
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
else
    echo "  indexed vector atomic       SKIP ($EXTLIB not built)"
fi

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "[smoke] some checks failed"
    exit 1
fi
echo "[smoke] passed"
