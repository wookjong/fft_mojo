#!/usr/bin/env bash
# =============================================================================
# Build the LLVM submodule for M²NDP backend work.
#
#   ./scripts/build-llvm.sh           # configure + build
#   ./scripts/build-llvm.sh check     # build, then run the RISC-V lit tests
#
# Only what backend work needs: the RISC-V target, the tools that consume the
# .ll that `./scripts/build.sh` produces, and lld. No clang, no other targets —
# that keeps a from-scratch build in the tens of minutes rather than hours.
# `check` additionally builds the tools the lit suite inspects object files
# with; see the tool list below.
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
    -DLLVM_ENABLE_PROJECTS="lld" \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_PARALLEL_LINK_JOBS="$LINK_JOBS" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

# Assertions are ON deliberately. Backend bugs surface as ISel and MachineInstr
# verifier assertions; without them the same bugs turn into silent miscompiles.

# lld is the one project built, and it earns its place: the linker has to
# understand the ISA string this LLVM emits. A distribution binutils does not
# -- ours rejects `zmmul` outright -- and that only shows up once objects from
# two toolchains are linked together, because merging the RISC-V attributes is
# what triggers the check. Discarding the section in the link script does not
# help; the merge happens first.

# llvm-lit is not in this list: cmake generates it as a script at configure
# time, so asking ninja for it fails with "unknown target".
TOOLS="llc llvm-mc opt llvm-as llvm-dis FileCheck count not lld"

# The lit suite needs more than backend work does, so these are built only for
# `check` -- the default build stays as small as the comment at the top claims.
# llvm-config is not optional: lit's configuration runs it to read the build
# mode, and without it the whole suite dies before running a single test. The
# rest are what the RISC-V tests inspect object files with; every one of the 31
# failures seen without them was the tool missing, not a codegen difference.
if [ "${1:-}" = "check" ]; then
    TOOLS="$TOOLS llvm-config llvm-objdump llvm-readobj llvm-readelf llvm-dwarfdump"
    # The MC suite needs a few more on top of what CodeGen does.
    TOOLS="$TOOLS llvm-nm yaml2obj split-file llvm-otool"
fi

# shellcheck disable=SC2086
ninja -C "$BUILD" -j "$JOBS" $TOOLS

echo ""
echo "[done] tools in $BUILD/bin"
"$BUILD/bin/llc" --version | grep -E 'LLVM version|riscv' || true

if [ "${1:-}" = "check" ]; then
    echo ""
    echo "[check] RISC-V lit tests"
    # MC as well as CodeGen: a vendor instruction is added encoding-first, and
    # the assembler and disassembler tests live under MC.
    "$BUILD/bin/llvm-lit" -sv \
        "$REPO/third_party/llvm-project/llvm/test/CodeGen/RISCV" \
        "$REPO/third_party/llvm-project/llvm/test/MC/RISCV"
fi
