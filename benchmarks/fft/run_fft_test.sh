#!/usr/bin/env bash
# =============================================================================
# Generate an M2NDP FFT Mojo kernel (make_fft_kernel.py) for one or more
# lengths and build + run each one against the real Mojo / M2NDP-Detour
# toolchain, checking the host-side reference-DFT self-check every generated
# kernel prints. Linux only (needs the real toolchain: mojo, our llc, Detour).
#
#   ./run_fft_test.sh 64                       # one length
#   ./run_fft_test.sh 8 16 64 160 320          # several, one build+run each
#   ./run_fft_test.sh 320 --inverse
#   ./run_fft_test.sh 64 --scratchpad-byte-budget 256
#   ./run_fft_test.sh 64 --compute-lanes 8      # opt back into the old,
#                                                # unchunked (spill-prone) shape
#   ./run_fft_test.sh 64 --keep -o /tmp/fftgen  # keep generated .mojo + build
#
# `--simd-lanes`/`--compute-lanes`/`--scratchpad-byte-budget`/`--tile-rows`/
# `--tile-cols` pass straight through to make_fft_kernel.py -- see its own
# --help for what each one means. In particular: `--simd-lanes` is the
# hardware launch granule (PooledRange/VECTOR_WIDTH in src/m2ndp.mojo) --
# do not use it to tune register pressure. `--compute-lanes` is the safe
# knob for that (default: make_fft_kernel.py's own min(simd_lanes, 4)).
#
# Toolchain discovery mirrors scripts/host-run.sh:
#   MOJO_ROOT     the Mojo install (default ./toolchain, via scripts/env.sh)
#   M2NDP_ROOT    repo root that has build/llvm/bin/llc and
#                 third_party/m2ndp-detour built (default: this repo's own
#                 root -- after ./scripts/setup.sh && ./scripts/build-llvm.sh).
#                 Inside the ghcr.io/psal-postech/mojo-m2ndp:main image, both
#                 already exist at /work, so run this from there, or:
#                   M2NDP_ROOT=/work ./run_fft_test.sh 64
#   LLC, RISCV_LD, M2NDP_DET, M2NDP_CONFIG   individually overridable too.
# =============================================================================
set -uo pipefail

FFT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$FFT_DIR/../.." && pwd)"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

M2NDP_ROOT="${M2NDP_ROOT:-$REPO}"
export M2NDP_ROOT
LLC="${LLC:-$M2NDP_ROOT/build/llvm/bin/llc}"
DET="${M2NDP_DET:-$M2NDP_ROOT/third_party/m2ndp-detour}"
export M2NDP_CONFIG="${M2NDP_CONFIG:-$DET/config/performance/M2NDP/m2ndp.config}"

# Colour only when stdout is a terminal; CI logs keep the words, drop the codes.
if [ -t 1 ]; then G='\033[32m'; R='\033[31m'; Y='\033[33m'; B='\033[1m'; D='\033[2m'; N='\033[0m'
else G=; R=; Y=; B=; D=; N=; fi

usage() {
    cat <<EOF
Usage: $(basename "$0") N [N ...] [options]

  N ...                       one or more FFT lengths to generate + test
  --inverse                   generate the inverse FFT
  --scratchpad-byte-budget B  bytes of scratchpad one leaf kernel may use
                               (default: 4096)
  --simd-lanes N               hardware launch granule (default: 8) --
                                do not change this to tune register pressure
  --compute-lanes N            SIMD width the emitted arithmetic actually
                                uses (default: make_fft_kernel.py's own
                                min(simd_lanes, 4))
  --tile-rows N / --tile-cols N  transpose tile size override
  --keep                       keep generated .mojo + build artifacts
  -o, --outdir DIR             directory for generated .mojo files
                                (default: a temp directory, removed unless
                                --keep is given)
  --timeout SECONDS            per-run simulator timeout (default: 180)
  -h, --help                    this message
EOF
}

LENGTHS=()
INVERSE=0
SCRATCHPAD_BUDGET=4096
SIMD_LANES=8
COMPUTE_LANES=""
TILE_ROWS=""
TILE_COLS=""
KEEP=0
OUTDIR=""
RUN_TIMEOUT=180

while [ $# -gt 0 ]; do
    case "$1" in
        --inverse) INVERSE=1; shift ;;
        --scratchpad-byte-budget) SCRATCHPAD_BUDGET="$2"; shift 2 ;;
        --simd-lanes) SIMD_LANES="$2"; shift 2 ;;
        --compute-lanes) COMPUTE_LANES="$2"; shift 2 ;;
        --tile-rows) TILE_ROWS="$2"; shift 2 ;;
        --tile-cols) TILE_COLS="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -o|--outdir) OUTDIR="$2"; shift 2 ;;
        --timeout) RUN_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) LENGTHS+=("$1"); shift ;;
    esac
done

if [ ${#LENGTHS[@]} -eq 0 ]; then
    echo "error: at least one FFT length is required" >&2
    usage >&2
    exit 2
fi

fail_setup() { printf "${R}error:${N} %s\n" "$1" >&2; exit 1; }

[ -x "$MOJO_BIN" ] || fail_setup "Mojo not found ($MOJO_BIN). Run ./scripts/setup.sh first."
[ -x "$LLC" ] || fail_setup "llc not found ($LLC). Run ./scripts/build-llvm.sh, or set LLC/M2NDP_ROOT."
[ -d "$DET" ] || fail_setup "Detour not found ($DET). Set M2NDP_DET or M2NDP_ROOT."
command -v cc >/dev/null || fail_setup "no C compiler (cc) found"

WORK="$(mktemp -d)"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

STAGE="$WORK/stage"
mkdir -p "$STAGE"
cp "$REPO"/src/*.mojo "$STAGE/"
cc -c -O2 "$REPO/sim/host_stubs.c" -o "$WORK/host_stubs.o" \
    || fail_setup "compiling sim/host_stubs.c failed"

if [ -n "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
    GEN_DIR="$OUTDIR"
else
    GEN_DIR="$WORK/generated"
    mkdir -p "$GEN_DIR"
fi

PASS_N=0
FAIL_N=0
SUMMARY=()

for LEN in "${LENGTHS[@]}"; do
    NAME="fft_fp32_N${LEN}"
    [ "$INVERSE" = 1 ] && NAME="${NAME}_inverse"
    MOJO_FILE="$GEN_DIR/${NAME}_generated.mojo"

    ARGS=("$LEN" --scratchpad-byte-budget "$SCRATCHPAD_BUDGET" --simd-lanes "$SIMD_LANES" -o "$MOJO_FILE")
    [ "$INVERSE" = 1 ] && ARGS+=(--inverse)
    [ -n "$COMPUTE_LANES" ] && ARGS+=(--compute-lanes "$COMPUTE_LANES")
    [ -n "$TILE_ROWS" ] && ARGS+=(--tile-rows "$TILE_ROWS")
    [ -n "$TILE_COLS" ] && ARGS+=(--tile-cols "$TILE_COLS")

    printf "${B}[fft-test]${N} N=%-8s${N} " "$LEN"

    if ! python3 "$FFT_DIR/make_fft_kernel.py" "${ARGS[@]}" >"$WORK/gen_${LEN}.log" 2>&1; then
        printf "${R}[GENERATE FAILED]${N}\n"
        sed 's/^/    /' "$WORK/gen_${LEN}.log"
        FAIL_N=$((FAIL_N + 1)); SUMMARY+=("N=$LEN: GENERATE FAILED"); continue
    fi

    cp "$MOJO_FILE" "$STAGE/"
    BASENAME="$(basename "$MOJO_FILE")"

    if ! ( cd "$STAGE" && timeout 600 "$MOJO_BIN" build "$BASENAME" \
            -o "$WORK/bin_${LEN}" -Xlinker "$WORK/host_stubs.o" -Xlinker -lm \
         ) >"$WORK/build_${LEN}.log" 2>&1; then
        printf "${R}[BUILD FAILED]${N}\n"
        grep -v -e tcmalloc -e Crashpad -e "abi()" -e "not a recog" "$WORK/build_${LEN}.log" \
            | sed 's/^/    /' | head -12
        FAIL_N=$((FAIL_N + 1)); SUMMARY+=("N=$LEN: BUILD FAILED"); continue
    fi

    RUN_LOG="$WORK/run_${LEN}.log"
    timeout "$RUN_TIMEOUT" "$WORK/bin_${LEN}" >"$RUN_LOG" 2>&1
    RC=$?

    # Each generated kernel is its own task.elf, linked into the simulator
    # lazily at launch time (not by the single `mojo build` above) -- a
    # spill warning is a diagnostic from *that* link step, so it lands in
    # RUN_LOG, never in build_${LEN}.log (confirmed: N=1024 baseline
    # without loop_stages spills a 464-byte frame that lowers to `vs1r.v`,
    # an instruction the simulator's decoder doesn't implement -- that
    # shows up only here, as "M2NDP kernel spills to memory" followed by
    # "[error] Unimplemented or Invalid Instruction" in RUN_LOG, while
    # build_${LEN}.log stays clean and this note used to never fire).
    SPILL_NOTE=""
    grep -qi "spills to memory" "$RUN_LOG" && SPILL_NOTE=" ${Y}(spill warning!)${N}"

    if [ "$RC" -ne 0 ]; then
        printf "${R}[SIMULATOR CRASHED]${N} (exit %s)%b\n" "$RC" "$SPILL_NOTE"
        tail -20 "$RUN_LOG" | sed 's/^/    /'
        FAIL_N=$((FAIL_N + 1)); SUMMARY+=("N=$LEN: CRASHED (exit $RC)"); continue
    fi

    if grep -q "verification passed" "$RUN_LOG"; then
        printf "${G}[PASS]${N}%b\n" "$SPILL_NOTE"
        PASS_N=$((PASS_N + 1)); SUMMARY+=("N=$LEN: PASS")
    elif grep -q "mismatch" "$RUN_LOG"; then
        printf "${R}[MISMATCH]${N}%b\n" "$SPILL_NOTE"
        grep -A3 "mismatch" "$RUN_LOG" | sed 's/^/    /'
        FAIL_N=$((FAIL_N + 1)); SUMMARY+=("N=$LEN: MISMATCH")
    else
        printf "${R}[UNKNOWN RESULT]${N}%b\n" "$SPILL_NOTE"
        tail -10 "$RUN_LOG" | sed 's/^/    /'
        FAIL_N=$((FAIL_N + 1)); SUMMARY+=("N=$LEN: UNKNOWN RESULT")
    fi
done

echo ""
printf "${B}===================== summary =====================${N}\n"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
printf "  ${G}%d passed${N}, ${R}%d failed${N}\n" "$PASS_N" "$FAIL_N"
if [ "$KEEP" = 1 ]; then
    echo "  kept build dir: $WORK"
    echo "  kept generated .mojo: $GEN_DIR"
fi

[ "$FAIL_N" -eq 0 ]
