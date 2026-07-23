#!/usr/bin/env bash
# =============================================================================
# Build Spike, the RISC-V ISA simulator, for M²NDP functional checking.
#
#   ./scripts/build-spike.sh          # configure + build
#   ./scripts/build-spike.sh check    # build, then run a smoke test
#
# Spike is what turns "the compiler emitted the instruction we meant" into
# "the instruction does what we meant". Nothing in this project has ever been
# executed; see docs/STATUS.md.
#
# Built out of tree, into build/spike, so the submodule stays clean.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SRC="$REPO/third_party/riscv-isa-sim"
BUILD="${SPIKE_BUILD_DIR:-$REPO/build/spike}"
PREFIX="${SPIKE_PREFIX:-$BUILD/install}"

if [ ! -f "$SRC/configure" ]; then
    echo "Spike submodule not checked out. Run:" >&2
    echo "    git submodule update --init --depth 1 third_party/riscv-isa-sim" >&2
    exit 1
fi

# configure hard-errors without it, and the message it prints does not say
# which package to install.
if ! command -v dtc >/dev/null; then
    echo "device-tree-compiler (dtc) not found; spike's configure requires it." >&2
    echo "    apt-get install -y device-tree-compiler" >&2
    exit 1
fi

JOBS="${JOBS:-$(nproc)}"

mkdir -p "$BUILD"
cd "$BUILD"

# Reconfigure only when it has not been done, so repeat builds are fast.
if [ ! -f "$BUILD/config.status" ]; then
    "$SRC/configure" --prefix="$PREFIX"
fi

make -j "$JOBS"
make install

# The M²NDP instructions are a loadable extension rather than a fork: Spike's
# extension interface reaches the vector unit (processor_t::VU is public), so
# nothing in the submodule needs patching. Built here because it links against
# the libriscv that was just installed.
#
# gnu++20 rather than the default: spike's own headers use C++20 features, and
# libstdc++ 11 needs <sys/syscall.h> pulled in for <atomic> under C++20 --
# which m2ndp_ext.cc does.
echo ""
echo "[build] m2ndp extension"
# The source tree is on the include path as well as the installed headers:
# decode_macros.h, which has the floating-point conversions and NaN boxing,
# is used internally by Spike and does not get installed.
g++ -std=gnu++20 -shared -fPIC -O2 \
    -o "$BUILD/libm2ndp_ext.so" "$REPO/sim/ext/m2ndp_ext.cc" \
    -I"$PREFIX/include" -I"$PREFIX/include/riscv" \
    -I"$PREFIX/include/fesvr" -I"$PREFIX/include/softfloat" \
    -I"$SRC/riscv" -I"$SRC/softfloat" -I"$SRC/fesvr" -I"$BUILD" \
    -L"$PREFIX/lib" -lriscv

echo ""
echo "[done] spike in $PREFIX/bin, extension in $BUILD/libm2ndp_ext.so"
"$PREFIX/bin/spike" --help 2>&1 | head -3 || true

if [ "${1:-}" = "check" ]; then
    echo ""
    echo "[check] smoke test"
    "$REPO/scripts/spike-smoke.sh"
fi
