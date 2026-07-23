#!/usr/bin/env bash
# =============================================================================
# Compile the benchmarks into LLVM IR / assembly under out/.
#
#   ./scripts/build.sh              # every benchmark, llvm + asm
#   ./scripts/build.sh spmv         # one benchmark only
#   EMISSION=asm ./scripts/build.sh # one emission only (llvm|asm)
#
# Benchmarks are built as whole modules: kernels are marked @export (nothing
# in the module calls them, so they would otherwise be dead code) and the
# target comes from the flags below. One file can hold several kernels, which
# `_compile_code` cannot do — it emits exactly one.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

if [ ! -x "$MOJO_BIN" ]; then
    echo "Mojo not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

mkdir -p out
TARGETS="${1:-}"
if [ -z "$TARGETS" ]; then
    TARGETS="$(ls benchmarks/*.mojo | xargs -n1 basename | sed 's/\.mojo$//')"
fi
EMISSIONS="${EMISSION:-llvm asm}"

# Target, matching m2ndp_target() in src/m2ndp.mojo.
TRIPLE="riscv64-unknown-elf"
CPU="generic-rv64"
FEATURES="+m,+a,+f,+d,+v,+zvl128b"

# Stage the benchmarks next to src/ in a temp dir so that `m2ndp` is on the
# module search path.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp src/*.mojo "$STAGE/"

FAILED=0
for name in $TARGETS; do
    wl="benchmarks/$name.mojo"
    [ -f "$wl" ] || { echo "!! $wl not found"; continue; }
    cp "$wl" "$STAGE/$name.mojo"
    for em in $EMISSIONS; do
        ext="ll"; [ "$em" = "asm" ] && ext="s"
        outfile="out/${name}.${ext}"
        printf "[build] %-12s %-4s -> %s\n" "$name" "$em" "$outfile"
        # Compile into a temp file first, so a failed build never destroys the
        # artifact from a previous successful run.
        ( cd "$STAGE" && timeout 180 "$MOJO_BIN" build \
            --target-triple "$TRIPLE" --target-cpu "$CPU" \
            --target-features "$FEATURES" --emit "$em" \
            "$name.mojo" -o "out.tmp" ) 2>"$STAGE/err.txt"
        if [ -s "$STAGE/out.tmp" ]; then
            mv "$STAGE/out.tmp" "$outfile"
        else
            echo "  !! failed (keeping any existing $outfile):"
            grep -v -e tcmalloc -e Crashpad -e "abi()" "$STAGE/err.txt" \
                | sed 's/^/     /' | head -5
            FAILED=1
        fi
    done
done

echo ""
echo "Generated files:"
ls -la out/ 2>/dev/null | tail -n +2

exit "$FAILED"
