#!/usr/bin/env bash
# =============================================================================
# Build the LLVM submodule for M²NDP backend work.
#
#   ./scripts/build-llvm.sh           # configure + build
#   ./scripts/build-llvm.sh check     # build, then run the RISC-V lit tests
#
# Only what backend work needs: the RISC-V target, and the tools that consume
# the .ll that `./scripts/build.sh` produces. No clang, no other targets —
# that keeps a from-scratch build in the tens of minutes rather than hours.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SRC="$REPO/third_party/llvm-project/llvm"
BUILD="${LLVM_BUILD_DIR:-$REPO/build/llvm}"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
    echo "LLVM submodule not checked out. Run:" >&2
    echo "    git submodule update --init --depth 1 third_party/llvm-project" >&2
    exit 1
fi

# Leave some headroom: linking LLVM is memory-hungry and will OOM long before
# compilation does, so link jobs are capped well below the compile width.
JOBS="${JOBS:-$(nproc)}"
LINK_JOBS="${LINK_JOBS:-$(( JOBS / 4 + 1 ))}"

cmake -G Ninja -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=RISCV \
    -DLLVM_ENABLE_PROJECTS="" \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_PARALLEL_LINK_JOBS="$LINK_JOBS" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

# Assertions are ON deliberately. Backend bugs surface as ISel and MachineInstr
# verifier assertions; without them the same bugs turn into silent miscompiles.

# llvm-lit is not in this list: cmake generates it as a script at configure
# time, so asking ninja for it fails with "unknown target".
ninja -C "$BUILD" -j "$JOBS" llc llvm-mc opt llvm-as llvm-dis FileCheck count not

echo ""
echo "[done] tools in $BUILD/bin"
"$BUILD/bin/llc" --version | grep -E 'LLVM version|riscv' || true

if [ "${1:-}" = "check" ]; then
    echo ""
    echo "[check] RISC-V lit tests"
    "$BUILD/bin/llvm-lit" -sv "$REPO/third_party/llvm-project/llvm/test/CodeGen/RISCV"
fi
