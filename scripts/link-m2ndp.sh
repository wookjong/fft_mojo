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
CFLAGS="-march=rv64g -mabi=lp64d -ffreestanding -nostdlib -fomit-frame-pointer -msmall-data-limit=0 -O2 -I$DET/src"

task="${1:?task.o}"
out="${2:?out.elf}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

"$GCC" $CFLAGS -c "$REPO/sim/m2ndp_launcher.c" -o "$tmp/launcher.o"
"$LLD" -T "$REPO/scripts/m2ndp.lds" -e _start "$tmp/launcher.o" "$task" -o "$out"
echo "linked $out"
