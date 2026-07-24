#!/usr/bin/env bash
# =============================================================================
# Emit each benchmark's device code into out/.
#
#   ./scripts/build.sh              # every benchmark
#   ./scripts/build.sh spmv         # one of them
#
# What comes out is what a launch compiles, not an approximation of it. A
# benchmark is one file -- kernels, device_main and the host main that launches
# them -- so the whole module cannot be compiled for the device: the host half
# is not device code and does not want to be. Instead each benchmark is built
# for the host and asked for its device IR:
#
#     ./spmv --emit-ir            # NDPTask.device_ir(), via compile_info
#
# which is exactly the text NDPTask.launch hands to llc.
#
# The assembly needs our llc, since only that one knows the vendor extension;
# without it the .ll files are still produced and the .s files are skipped.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

LLC="${LLC:-$REPO/build/llvm/bin/llc}"
FEATURES="+m,+a,+f,+d,+v,+zvl128b,+xm2ndp"

if [ ! -x "$MOJO_BIN" ]; then
    echo "Mojo not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

mkdir -p out
TARGETS="${*:-}"
if [ -z "$TARGETS" ]; then
    TARGETS="$(ls benchmarks/*.mojo | xargs -n1 basename | sed 's/\.mojo$//')"
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# The device symbols a host build needs to link. Nothing calls them here --
# --emit-ir returns before any launch -- but the linker still wants them.
cc -c -O2 sim/host_stubs.c -o "$OUT/host_stubs.o" || {
    echo "could not compile sim/host_stubs.c" >&2; exit 1; }

# Stage src/ beside the workload: Mojo's module search path is the directory
# being compiled in.
STAGE="$OUT/stage"
mkdir -p "$STAGE"
cp src/*.mojo "$STAGE/"

FAILED=0
for name in $TARGETS; do
    wl="benchmarks/$name.mojo"
    [ -f "$wl" ] || { echo "!! $wl not found"; FAILED=1; continue; }
    cp "$wl" "$STAGE/"

    printf "[build] %-14s " "$name"
    if ! ( cd "$STAGE" && timeout 600 "$MOJO_BIN" build "$name.mojo" \
            -o "$OUT/$name" -Xlinker "$OUT/host_stubs.o" ) 2>"$OUT/err"; then
        echo "FAIL (host build)"
        grep -v -e tcmalloc -e Crashpad -e "abi()" -e "not a recog" "$OUT/err" \
            | sed 's/^/    /' | head -8
        FAILED=1; continue
    fi

    # Into a temp file first, so a failed run never destroys a good artifact.
    if ! "$OUT/$name" --emit-ir > "$OUT/ir" 2>"$OUT/err" || [ ! -s "$OUT/ir" ]; then
        echo "FAIL (--emit-ir)"; sed 's/^/    /' "$OUT/err" | head -5
        FAILED=1; continue
    fi
    mv "$OUT/ir" "out/$name.ll"
    printf "llvm"

    if [ -x "$LLC" ]; then
        if "$LLC" -mtriple=riscv64-unknown-elf -mattr="$FEATURES" \
                "out/$name.ll" -o "$OUT/asm" 2>"$OUT/err"; then
            mv "$OUT/asm" "out/$name.s"
            printf " + asm"
        else
            printf " (asm failed)\n"; sed 's/^/    /' "$OUT/err" | head -3
            FAILED=1; continue
        fi
    fi
    echo ""
done

echo ""
echo "Generated files:"
ls -la out/ 2>/dev/null | tail -n +2

exit "$FAILED"
