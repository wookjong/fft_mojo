#!/usr/bin/env bash
# =============================================================================
# Link a task object (device_main + kernels) against the m2ndp controller
# launcher into a controller-runnable image with real addresses.
#
#   ./scripts/link-m2ndp.sh <task.o> <out.elf>
#
# The launcher (sim/m2ndp_launcher.c) is ours; the launch ABI header it includes
# is Detour's contract, taken from the m2ndp-detour tree (M2NDP_DET, default the
# submodule). Uses our ld.lld and scripts/m2ndp.lds.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LLD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"
DET="${M2NDP_DET:-$REPO/third_party/m2ndp-detour}"

# rv64g, not rv64gc: no compressed instructions, so the LLVMKernel disassembler
# (set up for the kernels' non-compressed code) can decode the launcher too.
# -fPIE -mcmodel=medany: the loader relocates the task into a pool code slot, so
# the launcher's code must be position-independent too, matching the medany the
# kernel object is built with; the image links -pie below.
CFLAGS="-march=rv64g -mabi=lp64d -ffreestanding -nostdlib -fomit-frame-pointer -msmall-data-limit=0 -O2 -fPIE -mcmodel=medany -I$DET/src -DM2NDP_ADDR_OFFSET=${M2NDP_ADDR_OFFSET:-0}ULL"

task="${1:?task.o}"
out="${2:?out.elf}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

"$GCC" $CFLAGS -c "$REPO/sim/m2ndp_launcher.c" -o "$tmp/launcher.o"
# -pie: a relocatable image the loader can place in a pool slot, not a fixed-
# address executable. Without it device_main's kernel address stays link-time.
"$LLD" -pie -T "$REPO/scripts/m2ndp.lds" -e _start "$tmp/launcher.o" "$task" -o "$out"
echo "linked $out"
