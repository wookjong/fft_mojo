#!/usr/bin/env bash
# =============================================================================
# Run a benchmark through the M²NDP-Detour timing simulator (ndp-controller).
#
#   ./scripts/timing-run.sh                 # vector_add
#   ./scripts/timing-run.sh vector_add
#
# Compiles the benchmark's device IR (out/<name>.ll) to a RISC-V/xm2ndp object,
# links it against the m2ndp launcher into a controller-runnable image, and runs
# it on Detour's controller core: device_main rings the doorbell, the kernels
# launch on the NDP units, and the harness checks the answer against the golden.
#
# A workload needs a Detour host harness (perf_runner/dev_launch*) that stages
# its inputs and checks its output; the case map below picks it by name.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

LLC="${LLC:-$REPO/build/llvm/bin/llc}"
# The Detour tree (source + build). CI points this at the image's prebuilt copy.
DET="${M2NDP_DET:-$REPO/third_party/m2ndp-detour}"
CONFIG="${M2NDP_CONFIG:-$DET/config/performance/M2NDP/m2ndp.config}"
FEATURES="+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp"

name="${1:-vector_add}"

case "$name" in
    vector_add) harness=dev_launch ;;
    loop_sum)   harness=dev_launch_loop ;;
    masked_add) harness=dev_launch_masked ;;
    *) echo "no Detour host harness for '$name' -- add one in $DET/perf_runner"; exit 1 ;;
esac

# Build the harness (and the sim it links) once.
if [ ! -x "$DET/build/bin/$harness" ]; then
    echo "[timing-run] building Detour sim ($harness)"
    cmake -S "$DET" -B "$DET/build" >/dev/null \
        && cmake --build "$DET/build" -j"$(nproc)" --target "$harness" >/dev/null \
        || { echo "sim build failed"; exit 1; }
fi

# Emit the benchmark's device IR if we do not already have it.
if [ ! -f "out/$name.ll" ]; then
    echo "[timing-run] emitting device IR (build.sh $name)"
    ./scripts/build.sh "$name" >/dev/null || { echo "build.sh $name failed"; exit 1; }
fi
[ -x "$LLC" ] || { echo "llc not found at $LLC -- run ./scripts/build-llvm.sh"; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "[timing-run] llc -> object"
# medany (LLVM's "medium"): the loader relocates the task into a pool code slot,
# so the kernel address device_main hands the doorbell has to be PC-relative
# (auipc). medlow bakes the link-time address and the controller finds no kernel.
"$LLC" -mtriple=riscv64 -mattr="$FEATURES" -code-model=medium -filetype=obj "out/$name.ll" -o "$work/$name.o" \
    || { echo "llc failed"; exit 1; }

echo "[timing-run] link against the launcher"
M2NDP_DET="$DET" "$REPO/scripts/link-m2ndp.sh" "$work/$name.o" "$work/$name.elf" >/dev/null \
    || { echo "link failed"; exit 1; }

echo "[timing-run] run on the controller"
"$DET/build/bin/$harness" "$work/$name.elf" "$CONFIG"
