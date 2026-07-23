#!/usr/bin/env bash
# =============================================================================
# Run the benchmarks under Spike and check the results.
#
#   ./scripts/simulate.sh                    # every benchmark, default topology
#   ./scripts/simulate.sh histogram          # one of them
#   CORES=4 CHUNK=1 ./scripts/simulate.sh    # a different spread over cores
#
# Inputs are generated on the host, the task reads and writes files, and the
# answer is compared here. Nothing about the data is compiled in, so changing
# the input does not mean rebuilding, and a mismatch can be looked at with
# ordinary tools.
#
# The topology is an argument rather than a constant because the answer must
# not depend on it. See sim/topology.h.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SPIKE="${SPIKE:-$REPO/build/spike/install/bin/spike}"
EXTLIB="${EXTLIB:-$REPO/build/spike/libm2ndp_ext.so}"
LLC="${LLC:-$REPO/build/llvm/bin/llc}"
LD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"
OBJDUMP="${OBJDUMP:-$REPO/build/llvm/bin/llvm-objdump}"
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"

export LD_LIBRARY_PATH="$REPO/build/spike/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ISA="rv64gcv_zvl128b"
FEATURES="+m,+a,+f,+d,+v,+zvl128b,+xm2ndp"
# scripts/m2ndp.lds puts code at 0x10000, below where Spike puts memory by
# default. The region stops short of 0x2000000, where Spike's CLINT lives.
MEM="-m0x10000:0x1ff0000"

# No small data: it needs gp, which nothing sets up here, and the sections it
# would create have no home in a bare-metal layout.
CFLAGS="-march=rv64gc -mabi=lp64d -ffreestanding -nostdlib -fomit-frame-pointer
        -msmall-data-limit=0 -O2 -Isim"

CORES="${CORES:-1}"
CHUNK="${CHUNK:-1}"

BENCHES="${*:-vector_add histogram}"

fail() { echo "  $*"; exit 1; }

# A panic prints mepc; turning that into instructions is the host's job, since
# it has the disassembler and knows the vendor extension. A target-side dump
# could only manage raw words.
show_panic() {
    local log="$1" elf="$2"
    sed 's/^/    /' "$log"
    local epc
    epc=$(sed -n 's/.*mepc  *0x0*\([0-9a-f][0-9a-f]*\).*/\1/p' "$log" | head -1)
    [ -n "$epc" ] || return 0
    [ -x "$OBJDUMP" ] || return 0
    echo "    around mepc:"
    "$OBJDUMP" -d --mattr="$FEATURES" "$elf" 2>/dev/null |
        grep -n "^ *$epc:" | head -1 | cut -d: -f1 | {
            read -r line || return 0
            "$OBJDUMP" -d --mattr="$FEATURES" "$elf" 2>/dev/null |
                sed -n "$((line > 8 ? line - 8 : 1)),$((line + 8))p" |
                sed "s/^/      /;s/^      \( *$epc:\)/   -> \1/"
        }
}

for t in "$SPIKE" "$LLC" "$LD"; do
    [ -x "$t" ] || fail "$t not found. Run ./scripts/build-llvm.sh and ./scripts/build-spike.sh"
done
command -v "$GCC" >/dev/null || \
    fail "$GCC not found. apt-get install -y gcc-riscv64-unknown-elf"
[ -f "$EXTLIB" ] || fail "$EXTLIB not built. Run ./scripts/build-spike.sh"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Everything that is not the kernel: the launcher is common, and a benchmark
# only describes itself.
COMMON=""
for c in sim/start.c sim/htif.c sim/panic.c sim/launcher.c; do
    o="$OUT/$(basename "$c" .c).o"
    # shellcheck disable=SC2086
    "$GCC" $CFLAGS -c "$c" -o "$o" || fail "compiling $c"
    COMMON="$COMMON $o"
done

echo "[simulate] cores=$CORES chunk=$CHUNK"
FAILED=0

for bench in $BENCHES; do
    printf "  %-14s " "$bench"

    if [ ! -f "out/$bench.ll" ]; then
        echo "SKIP (run ./scripts/build.sh)"
        continue
    fi
    if [ ! -f "sim/bench/$bench.c" ]; then
        echo "SKIP (no sim/bench/$bench.c)"
        continue
    fi

    # shellcheck disable=SC2086
    if ! "$GCC" $CFLAGS -c "sim/bench/$bench.c" -o "$OUT/b.o" 2>"$OUT/err"; then
        echo "FAIL (launcher)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
    fi
    if ! "$LLC" -mtriple=riscv64-unknown-elf -mattr="$FEATURES" -filetype=obj \
            "out/$bench.ll" -o "$OUT/k.o" 2>"$OUT/err"; then
        echo "FAIL (llc)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
    fi
    # shellcheck disable=SC2086
    if ! "$LD" -T scripts/m2ndp.lds -e _start $COMMON "$OUT/b.o" "$OUT/k.o" \
            -o "$OUT/t.elf" 2>"$OUT/err"; then
        echo "FAIL (link)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
    fi

    if ! python3 sim/gen-input.py "$bench" "$CORES" "$OUT" > "$OUT/files" \
            2>"$OUT/err"; then
        echo "FAIL (generating input)"; sed 's/^/    /' "$OUT/err"
        FAILED=1; continue
    fi
    read -r -a FILES < "$OUT/files"

    timeout 600 "$SPIKE" $MEM --extlib="$EXTLIB" --extension=m2ndp \
        --isa="$ISA" "$OUT/t.elf" "$CORES" "$CHUNK" "${FILES[@]}" \
        > "$OUT/log" 2>&1
    rc=$?

    case "$rc" in
      0) ;;
      3) echo "FAIL (panic in the target)"; show_panic "$OUT/log" "$OUT/t.elf"
         FAILED=1; continue ;;
      124) echo "FAIL (timed out)"; FAILED=1; continue ;;
      *) echo "FAIL (exit $rc)"; sed 's/^/    /' "$OUT/log"; FAILED=1; continue ;;
    esac

    if ! python3 sim/gen-input.py --check "$bench" "$OUT" >"$OUT/err" 2>&1; then
        echo "FAIL (wrong answer)"; sed 's/^/    /' "$OUT/err"; FAILED=1; continue
    fi
    echo "OK"
done

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "[simulate] some benchmarks failed"
    exit 1
fi
echo "[simulate] passed"
