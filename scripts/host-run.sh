#!/usr/bin/env bash
# =============================================================================
# Run a workload from a host program written in Mojo.
#
#   ./scripts/host-run.sh                    # every benchmark, default topology
#   ./scripts/host-run.sh vector_add
#   ./scripts/host-run.sh histogram 4 256    # cores stride(bytes)
#
# A benchmark is one file holding its kernels, its device_main and the host
# main that launches them and checks the answer -- single source, the way a
# .cu is. This script only builds the two things that file cannot make for
# itself: the device-side launcher, and its own binary linked against the
# device-symbol stubs.
#
# What launch_task needs from the environment:
#   M2NDP_ROOT        the repo, to find llc, the linker and the simulator
#   M2NDP_COMMON_OBJ  the pre-built device launcher
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

LLC="${LLC:-$REPO/build/llvm/bin/llc}"
LD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"
SPIKE="${SPIKE:-$REPO/build/spike/install/bin/spike}"
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"

export LD_LIBRARY_PATH="$REPO/build/spike/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CFLAGS="-march=rv64gc -mabi=lp64d -ffreestanding -nostdlib -fomit-frame-pointer
        -msmall-data-limit=0 -O2 -Isim"

# With no benchmark named, run each in turn -- whatever is in benchmarks/,
# the same way build.sh finds them.
if [ $# -eq 0 ]; then
    rc=0
    for b in $(ls benchmarks/*.mojo | xargs -n1 basename | sed 's/\.mojo$//'); do
        "$0" "$b" || rc=1
    done
    exit "$rc"
fi

bench="${1:-vector_add}"
cores="${2:-${CORES:-1}}"
stride="${3:-${STRIDE:-256}}"

fail() { echo "  $*"; exit 1; }

[ -x "$MOJO_BIN" ] || fail "Mojo not found. Run ./scripts/setup.sh first."
for t in "$SPIKE" "$LLC" "$LD"; do
    [ -x "$t" ] || fail "$t not found. Run ./scripts/build-llvm.sh and ./scripts/build-spike.sh"
done
command -v "$GCC" >/dev/null || fail "$GCC not found"
[ -f "benchmarks/$bench.mojo" ] || fail "no benchmarks/$bench.mojo"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "[host-run] $bench, cores=$cores stride=$stride"

# The device-side launcher, once. The task itself is compiled by the host
# program at launch; this is only the machine's half -- start, htif, panic and
# the launcher -- relocatably linked into one object the launcher expects.
# shellcheck disable=SC2086
"$GCC" $CFLAGS -r sim/start.c sim/htif.c sim/panic.c sim/launcher.c \
    -o "$OUT/common.o" || fail "compiling the launcher"

# The device symbols, so the host program links. See sim/host_stubs.c.
cc -c -O2 sim/host_stubs.c -o "$OUT/host_stubs.o" || fail "compiling host_stubs.c"

# Stage src/ and the workload together, because Mojo's module search path is
# the directory being compiled in. One file: the workload holds its kernels,
# its device_main and the host main that launches them, the way a .cu does.
STAGE="$OUT/stage"
mkdir -p "$STAGE"
cp src/*.mojo "$STAGE/"
cp "benchmarks/$bench.mojo" "$STAGE/"

( cd "$STAGE" && timeout 600 "$MOJO_BIN" build "$bench.mojo" \
    -o "$OUT/host" -Xlinker "$OUT/host_stubs.o" ) 2>"$OUT/err" \
    || { echo "  building the host program failed:"
         grep -v -e tcmalloc -e Crashpad -e "abi()" -e "not a recog" "$OUT/err" \
             | sed 's/^/    /' | head -12; exit 1; }

# The machine is config, not an argument: the runtime reads a description and
# configures itself. Varying it is varying the file, which is what checking
# that an answer does not depend on the hardware actually means.
# Only cores and stride vary per run; the packet and the pool come from
# the checked-in description.
{ echo "cores = $cores"; echo "stride = $stride"
  grep -E '^[[:space:]]*(packet|pool_base|pool_bytes)[[:space:]]*=' config/machine.conf
} > "$OUT/machine.conf"

export M2NDP_ROOT="$REPO"
export M2NDP_COMMON_OBJ="$OUT/common.o"
export M2NDP_MACHINE_CONFIG="$OUT/machine.conf"

"$OUT/host"
