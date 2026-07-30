#!/usr/bin/env bash
# Build a Mojo workload's host program and run it.
#   ./scripts/host-run.sh [benchmark]   # no name -> every benchmark
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

LLC="${LLC:-$REPO/build/llvm/bin/llc}"
LD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"

# No benchmark named: run each in benchmarks/ in turn.
if [ $# -eq 0 ]; then
    rc=0
    for b in $(ls benchmarks/*.mojo | xargs -n1 basename | sed 's/\.mojo$//'); do
        "$0" "$b" || rc=1
    done
    exit "$rc"
fi

bench="${1:-vector_add}"

fail() { echo "  $*"; exit 1; }

[ -x "$MOJO_BIN" ] || fail "Mojo not found. Run ./scripts/setup.sh first."
for t in "$LLC" "$LD"; do
    [ -x "$t" ] || fail "$t not found. Run ./scripts/build-llvm.sh"
done
command -v "$GCC" >/dev/null || fail "$GCC not found"
[ -f "benchmarks/$bench.mojo" ] || fail "no benchmarks/$bench.mojo"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "[host-run] $bench"

# Device-symbol stubs, so the host program links.
cc -c -O2 sim/host_stubs.c -o "$OUT/host_stubs.o" || fail "compiling host_stubs.c"

# Mojo's module search path is the compile directory, so stage src/ with the workload.
STAGE="$OUT/stage"
mkdir -p "$STAGE"
cp src/*.mojo "$STAGE/"
cp "benchmarks/$bench.mojo" "$STAGE/"

( cd "$STAGE" && timeout 600 "$MOJO_BIN" build "$bench.mojo" \
    -o "$OUT/host" -Xlinker "$OUT/host_stubs.o" ) 2>"$OUT/err" \
    || { echo "  building the host program failed:"
         grep -v -e tcmalloc -e Crashpad -e "abi()" -e "not a recog" "$OUT/err" \
             | sed 's/^/    /' | head -12; exit 1; }

# The machine is config, not an argument: the runtime and the device read one
# simulator description and configure themselves from it. Point M2NDP_CONFIG at
# another to model a different machine.
export M2NDP_ROOT="$REPO"
export M2NDP_CONFIG="${M2NDP_CONFIG:-$REPO/third_party/m2ndp-detour/config/performance/M2NDP/m2ndp.config}"

"$OUT/host"
