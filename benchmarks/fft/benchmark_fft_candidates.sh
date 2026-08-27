#!/usr/bin/env bash
# =============================================================================
# Build + run every candidate planning.fft_plan_search.generate_candidates(N)
# finds (via make_fft_kernel.py --plan-index), against the real M2NDP
# simulator, with --no-reference-check (see run_fft_test.sh's own --no-
# reference-check help). Compares each candidate's actual simulated cycle
# count against fft_cost_model.estimate_cost's own prediction, ranked side
# by side -- the point of this script is validating/calibrating that cost
# model against real simulator behavior, not correctness: --no-reference-
# check means no PASS/FAIL check runs here at all. Verify correctness
# separately (run_fft_test.sh at the same N with the reference check left
# on -- fine up to the N where its own O(N^2) host DFT stops finishing in a
# reasonable time; past that, the Python-side numpy.fft harness this
# script's own --no-reference-check markers are meant for).
#
#   ./benchmark_fft_candidates.sh 4096
#   ./benchmark_fft_candidates.sh 16384 --inverse --max-candidates 6
#   ./benchmark_fft_candidates.sh 1024 --keep -o /tmp/fftbench
#
# Toolchain discovery and most options mirror run_fft_test.sh; see its own
# --help for what each passthrough one means.
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

if [ -t 1 ]; then G='\033[32m'; R='\033[31m'; Y='\033[33m'; B='\033[1m'; D='\033[2m'; N='\033[0m'
else G=; R=; Y=; B=; D=; N=; fi

usage() {
    cat <<EOF
Usage: $(basename "$0") N [options]

  N                            the single FFT length to compare candidates
                                for (planning.fft_plan_search.generate_
                                candidates(N) -- one N per run, unlike
                                run_fft_test.sh's several)
  --inverse                    generate the inverse FFT
  --scratchpad-byte-budget B  bytes of scratchpad one leaf kernel may use
                               (default: 4096) -- see make_fft_kernel.py
  --simd-lanes N                hardware launch granule -- see
                                make_fft_kernel.py --help (default: 8)
  --compute-lanes N             SIMD width the emitted arithmetic actually
                                uses -- see make_fft_kernel.py --help
  --batch B                     how many independent length-N transforms
                                to run in one launch (default: 1) -- see
                                make_fft_kernel.py --help. Applied to
                                every candidate the same way; not itself
                                a ranked search axis.
  --max-candidates K            cap how many ranked candidates to actually
                                build+run (default: 8 -- generate_candidates
                                can return up to 24; each one is a real
                                build + simulator run, so this bounds
                                wall-clock). All candidates still get
                                dumped and shown in the estimated_cost
                                ranking; only the top K by estimated_cost
                                get built and run.
  --keep                       keep generated .mojo + build artifacts
  -o, --outdir DIR             directory for generated .mojo files
                                (default: a temp directory, removed unless
                                --keep is given)
  --timeout SECONDS            per-run simulator timeout (default: 300 --
                                higher than run_fft_test.sh's default since
                                this script exists for large-N runs)
  -h, --help                    this message
EOF
}

LEN=""
INVERSE=0
SCRATCHPAD_BUDGET=4096
SIMD_LANES=8
COMPUTE_LANES=""
BATCH=1
MAX_CANDIDATES=8
KEEP=0
OUTDIR=""
RUN_TIMEOUT=300

while [ $# -gt 0 ]; do
    case "$1" in
        --inverse) INVERSE=1; shift ;;
        --scratchpad-byte-budget) SCRATCHPAD_BUDGET="$2"; shift 2 ;;
        --simd-lanes) SIMD_LANES="$2"; shift 2 ;;
        --compute-lanes) COMPUTE_LANES="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --max-candidates) MAX_CANDIDATES="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        -o|--outdir) OUTDIR="$2"; shift 2 ;;
        --timeout) RUN_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [ -n "$LEN" ]; then
                echo "error: only one N is supported (got a second: $1)" >&2
                usage >&2; exit 2
            fi
            LEN="$1"; shift ;;
    esac
done

if [ -z "$LEN" ]; then
    echo "error: an FFT length is required" >&2
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

# ---- 1. dump every candidate's own estimated_cost (no build yet) ----------

DUMP_ARGS=("$LEN" --scratchpad-byte-budget "$SCRATCHPAD_BUDGET" --simd-lanes "$SIMD_LANES" --batch "$BATCH" --dump-candidates)
[ "$INVERSE" = 1 ] && DUMP_ARGS+=(--inverse)
[ -n "$COMPUTE_LANES" ] && DUMP_ARGS+=(--compute-lanes "$COMPUTE_LANES")

DUMP_LOG="$WORK/dump_candidates.log"
if ! python3 "$FFT_DIR/make_fft_kernel.py" "${DUMP_ARGS[@]}" >"$DUMP_LOG" 2>&1; then
    fail_setup "--dump-candidates failed:
$(sed 's/^/    /' "$DUMP_LOG")"
fi

# Parsed straight out of format_plan_summary's own text (fft_plan_search.py)
# -- not a separate machine-readable format, so this stays in lockstep with
# --dump-candidates's own output by construction (both come from the same
# function). Already rank-ordered by estimated_cost (rank_candidates runs
# before format_plan_summary in make_fft_kernel.py's --dump-candidates path).
mapfile -t CAND_INDICES < <(grep -oP '^\[\K[0-9]+(?=\])' "$DUMP_LOG")
mapfile -t CAND_COSTS < <(grep -oP 'estimated_cost\s*=\s*\K[0-9.]+' "$DUMP_LOG")
mapfile -t CAND_CHOICES < <(grep -oP '^\s*choices:\s*\K.*' "$DUMP_LOG")

TOTAL_CANDIDATES=${#CAND_INDICES[@]}
[ "$TOTAL_CANDIDATES" -gt 0 ] || fail_setup "no candidates parsed from --dump-candidates output ($DUMP_LOG)"

RUN_COUNT=$MAX_CANDIDATES
[ "$RUN_COUNT" -gt "$TOTAL_CANDIDATES" ] && RUN_COUNT=$TOTAL_CANDIDATES

printf "${B}[benchmark]${N} N=%s: %d candidate(s) total, building+running the top %d by estimated_cost\n\n" \
    "$LEN" "$TOTAL_CANDIDATES" "$RUN_COUNT"

# ---- 2. build + run each of the top RUN_COUNT candidates -------------------

RESULTS=()  # "index cost cycles_or_FAILED choices..."
FAIL_N=0

for ((i = 0; i < RUN_COUNT; i++)); do
    K="${CAND_INDICES[$i]}"
    COST="${CAND_COSTS[$i]}"
    CHOICES="${CAND_CHOICES[$i]}"
    NAME="fft_fp32_N${LEN}"
    [ "$INVERSE" = 1 ] && NAME="${NAME}_inverse"
    NAME="${NAME}_plan${K}"
    MOJO_FILE="$GEN_DIR/${NAME}_generated.mojo"

    ARGS=("$LEN" --scratchpad-byte-budget "$SCRATCHPAD_BUDGET" --simd-lanes "$SIMD_LANES"
          --batch "$BATCH" --plan-index "$K" --no-reference-check -o "$MOJO_FILE")
    [ "$INVERSE" = 1 ] && ARGS+=(--inverse)
    [ -n "$COMPUTE_LANES" ] && ARGS+=(--compute-lanes "$COMPUTE_LANES")

    printf "${B}[plan %s]${N} cost=%-12s %s ... " "$K" "$COST" "$CHOICES"

    if ! python3 "$FFT_DIR/make_fft_kernel.py" "${ARGS[@]}" >"$WORK/gen_${K}.log" 2>&1; then
        printf "${R}[GENERATE FAILED]${N}\n"
        sed 's/^/    /' "$WORK/gen_${K}.log"
        FAIL_N=$((FAIL_N + 1)); RESULTS+=("$K $COST FAILED $CHOICES"); continue
    fi

    cp "$MOJO_FILE" "$STAGE/"
    BASENAME="$(basename "$MOJO_FILE")"

    if ! ( cd "$STAGE" && timeout 600 "$MOJO_BIN" build "$BASENAME" \
            -o "$WORK/bin_${K}" -Xlinker "$WORK/host_stubs.o" -Xlinker -lm \
         ) >"$WORK/build_${K}.log" 2>&1; then
        printf "${R}[BUILD FAILED]${N}\n"
        grep -v -e tcmalloc -e Crashpad -e "abi()" -e "not a recog" "$WORK/build_${K}.log" \
            | sed 's/^/    /' | head -12
        FAIL_N=$((FAIL_N + 1)); RESULTS+=("$K $COST FAILED $CHOICES"); continue
    fi

    RUN_LOG="$WORK/run_${K}.log"
    timeout "$RUN_TIMEOUT" "$WORK/bin_${K}" >"$RUN_LOG" 2>&1
    RC=$?

    # Same spill diagnostic run_fft_test.sh checks -- see its own comment
    # on why this only ever shows up in the run log, never the build log.
    SPILL_NOTE=""
    grep -qi "spills to memory" "$RUN_LOG" && SPILL_NOTE=" ${Y}(spill warning!)${N}"

    if [ "$RC" -ne 0 ]; then
        printf "${R}[SIMULATOR CRASHED]${N} (exit %s)%b\n" "$RC" "$SPILL_NOTE"
        tail -20 "$RUN_LOG" | sed 's/^/    /'
        FAIL_N=$((FAIL_N + 1)); RESULTS+=("$K $COST FAILED $CHOICES"); continue
    fi

    # The simulator's own "Gantt info: ... finished NDP kernel ..." lines
    # print each launched kernel's completion in "ndp cycle" terms; kernels
    # in one recursive plan launch strictly one after another from a single
    # host main() (confirmed in a real run log: launch id k+1's own "Gate
    # info: ... launched ..." line always comes after launch id k's
    # "finished" line), so the *last* one's ndp cycle is the whole plan's
    # own total simulated device time.
    CYCLES="$(grep "Gantt info:.*finished NDP kernel" "$RUN_LOG" | grep -oP 'ndp cycle \K[0-9]+' | tail -1)"
    if [ -z "$CYCLES" ]; then
        printf "${R}[NO CYCLE COUNT FOUND]${N}%b\n" "$SPILL_NOTE"
        tail -10 "$RUN_LOG" | sed 's/^/    /'
        FAIL_N=$((FAIL_N + 1)); RESULTS+=("$K $COST FAILED $CHOICES"); continue
    fi

    printf "${G}%s ndp cycles${N}%b\n" "$CYCLES" "$SPILL_NOTE"
    RESULTS+=("$K $COST $CYCLES $CHOICES")
done

# ---- 3. side-by-side table + does the cost model's own pick agree? --------

echo ""
printf "${B}===================== N=%s: estimated_cost vs. actual ndp_cycles =====================${N}\n" "$LEN"
printf "  %-5s %-14s %-12s %s\n" "plan" "estimated_cost" "ndp_cycles" "choices"
for line in "${RESULTS[@]}"; do
    read -r K COST CYCLES CHOICES <<< "$line"
    printf "  %-5s %-14s %-12s %s\n" "$K" "$COST" "$CYCLES" "$CHOICES"
done

BEST_BY_COST=""
BEST_BY_COST_VAL=""
BEST_BY_CYCLES=""
BEST_BY_CYCLES_VAL=""
for line in "${RESULTS[@]}"; do
    read -r K COST CYCLES _ <<< "$line"
    [ "$CYCLES" = "FAILED" ] && continue
    if [ -z "$BEST_BY_COST_VAL" ] || python3 -c "import sys; sys.exit(0 if $COST < $BEST_BY_COST_VAL else 1)"; then
        BEST_BY_COST="$K"; BEST_BY_COST_VAL="$COST"
    fi
    if [ -z "$BEST_BY_CYCLES_VAL" ] || [ "$CYCLES" -lt "$BEST_BY_CYCLES_VAL" ]; then
        BEST_BY_CYCLES="$K"; BEST_BY_CYCLES_VAL="$CYCLES"
    fi
done

echo ""
if [ -n "$BEST_BY_COST" ]; then
    printf "  cost model's pick: plan %s (estimated_cost=%s)\n" "$BEST_BY_COST" "$BEST_BY_COST_VAL"
    printf "  actual best:       plan %s (%s ndp cycles)\n" "$BEST_BY_CYCLES" "$BEST_BY_CYCLES_VAL"
    if [ "$BEST_BY_COST" = "$BEST_BY_CYCLES" ]; then
        printf "  ${G}agree${N}\n"
    else
        printf "  ${Y}disagree -- fft_cost_model.CostWeights may need retuning for this N${N}\n"
    fi
else
    printf "  ${R}no candidate finished a run -- nothing to compare${N}\n"
fi

if [ "$KEEP" = 1 ]; then
    echo "  kept build dir: $WORK"
    echo "  kept generated .mojo: $GEN_DIR"
fi

[ "$FAIL_N" -eq 0 ]
