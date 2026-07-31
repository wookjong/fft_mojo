#!/usr/bin/env bash
# Run a workload against an AddressSanitizer build of the simulator.
#   scripts/asan-run.sh [benchmark]
#
# Build the sim once with ASAN first:
#   cmake -S third_party/m2ndp-detour -B third_party/m2ndp-detour/build -DM2NDP_ASAN=ON
#   cmake --build third_party/m2ndp-detour/build -j --target m2ndp_run
# That build shifts the address map by this offset (into ASAN's HighMem); the host
# and launcher pick it up here so all three agree. allow_user_segv_handler=0 hands
# SEGV to ASAN instead of the panic diagnostics so faults get a report.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

export M2NDP_ADDR_OFFSET="${M2NDP_ADDR_OFFSET:-0x200000000000}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:handle_segv=1:allow_user_segv_handler=0:halt_on_error=1:abort_on_error=0}"

exec "$REPO/scripts/host-run.sh" "$@"
