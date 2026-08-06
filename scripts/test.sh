#!/usr/bin/env bash
# =============================================================================
# The one entry point for the test hierarchy. Runs the tiers in order, cheapest
# first, and stops at the first that fails. Each check prints one line saying
# what it verified; a [FAIL] carries the reason right under it. The simulator's
# own output is swallowed, so a result reads at a glance. Cases come from
# tests/manifest.tsv.
#
#   ./scripts/test.sh                 # every tier, then the coverage report
#   ./scripts/test.sh t0 t1           # only these, in this order
#   ./scripts/test.sh coverage        # just the opcode-coverage map
#   REQUIRE_ALL_GREEN=1 ./scripts/test.sh   # require every case, xfail included
#
# Tiers (cheapest first):
#   t0  toolchain   the pinned LLVM can build M2NDP device code
#   t1  codegen     each workload lowers to the device code we expect (static)
#   t2  controller  device_main runs on the controller and moves data
#   t3  basic-op    one-operation kernels on Detour (pending)
#   t4  workload    a whole workload launches and computes correctly (E2E)
#   coverage        the RVV/scalar/CSR/M2NDP opcodes each workload exercises
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MANIFEST="${MANIFEST:-$REPO/tests/manifest.tsv}"
LLC="${LLC:-$REPO/build/llvm/bin/llc}"
LD="${RISCV_LD:-$REPO/build/llvm/bin/ld.lld}"

# Colour only when stdout is a terminal; CI logs keep the words, drop the codes.
if [ -t 1 ]; then G='\033[32m'; R='\033[31m'; Y='\033[33m'; B='\033[1m'; D='\033[2m'; N='\033[0m'
else G=; R=; Y=; B=; D=; N=; fi

FAIL=0; PASS_N=0; FAIL_N=0; SKIP_N=0; XFAIL_N=0; XPASS_N=0
section() { printf "\n${B}[%s]${N}\n" "$1"; }
pass()    { printf "  ${G}[PASS]${N} %s\n" "$1"; PASS_N=$((PASS_N + 1)); }
skip()    { printf "  ${Y}[SKIP]${N} %s\n" "$1"; SKIP_N=$((SKIP_N + 1)); }
# fail "<what>"  or  fail "<what>" "<detail line>"
fail()    { printf "  ${R}[FAIL]${N} %s\n" "$1"; [ -n "${2:-}" ] && printf "         ${D}%s${N}\n" "$2"; FAIL=1; FAIL_N=$((FAIL_N + 1)); }
# A case the manifest marks as known-broken: its failure is expected and does not
# fail the run; a pass is an [XPASS] to promote out of xfail.
xfail()   { printf "  ${Y}[XFAIL]${N} %s\n" "$1"; [ -n "${2:-}" ] && printf "          ${D}%s${N}\n" "$2"; XFAIL_N=$((XFAIL_N + 1)); }
xpass()   { printf "  ${Y}[XPASS]${N} %s ${D}(now passes -- promote it out of xfail)${N}\n" "$1"; XPASS_N=$((XPASS_N + 1)); }
summary() {
    local total=$((PASS_N + FAIL_N + SKIP_N + XFAIL_N + XPASS_N))
    printf "\n${B}Summary:${N} ${G}%d passed${N}, ${R}%d failed${N}, ${Y}%d skipped${N}" "$PASS_N" "$FAIL_N" "$SKIP_N"
    [ "$XFAIL_N" -gt 0 ] && printf ", ${Y}%d xfail${N}" "$XFAIL_N"
    [ "$XPASS_N" -gt 0 ] && printf ", ${Y}%d xpass${N}" "$XPASS_N"
    printf " ${D}(%d checks total)${N}\n" "$total"
    if [ "$REQUIRE_ALL_GREEN" = 1 ]; then
        printf "${Y}REQUIRE_ALL_GREEN=1${N} — every case must pass; xfail does not excuse a failure\n"
    fi
}

# The case names in a tier: column 1 where column 2 (comma list) contains the tier.
manifest_names() {
    awk -v t="$1" -F'\t' '
        /^[[:space:]]*#/ || NF < 2 { next }
        { n = split($2, ts, ","); for (i = 1; i <= n; i++) if (ts[i] == t) print $1 }
    ' "$MANIFEST" 2>/dev/null
}

# A named column for a case: manifest_field <name> <col>. Columns are
#   1 name  2 tiers  3 family  4 check  5 expect  6 note
manifest_field() {
    awk -v name="$1" -v col="$2" -F'\t' '
        /^[[:space:]]*#/ || NF < 2 { next }
        $1 == name { print $col; exit }
    ' "$MANIFEST" 2>/dev/null
}
manifest_check()  { manifest_field "$1" 4; }
manifest_expect() { manifest_field "$1" 5; }
manifest_note()   { manifest_field "$1" 6; }

# The one switch that flips the suite to all-green: with REQUIRE_ALL_GREEN=1
# every case must pass and the manifest's "xfail" no longer excuses a failure.
# Off by default, so today's known-broken workloads report [XFAIL], not [FAIL].
REQUIRE_ALL_GREEN="${REQUIRE_ALL_GREEN:-0}"

tier_t0() {
    section "Toolchain — the pinned LLVM builds M2NDP device code"
    [ -x "$LLC" ] || { fail "llc is present and runnable" "not executable at $LLC"; return; }
    local ver; ver="$("$LLC" --version 2>/dev/null | sed -n 's/.*LLVM version \([^ ]*\).*/\1/p' | head -1)"
    pass "llc runs — the device compiler${ver:+ (LLVM $ver)}"
    [ -x "$LD" ] && pass "ld.lld runs — the task linker" || fail "ld.lld is present and runnable" "not executable at $LD"
    "$LLC" --version 2>/dev/null | grep -q 'riscv64' \
        && pass "riscv64 is a registered llc target" || fail "llc knows the riscv64 target"
    if "$LLC" -mtriple=riscv64-unknown-elf -mattr=help 2>&1 | grep -qi 'xm2ndp'; then
        pass "the xm2ndp vendor extension is available"
    else
        fail "llc has the xm2ndp extension"
        _t0_why_no_xm2ndp
    fi
}

# The above fails in CI now and then and passes on a re-run, with the same image
# digest and an llc whose hash matches the release asset -- so the binary is not
# the variable. This says which llc ran, what it linked, and whether the feature
# list came out whole, since a truncated one would explain the miss.
_t0_why_no_xm2ndp() {
    local h; h="$(mktemp)"
    "$LLC" -mtriple=riscv64-unknown-elf -mattr=help >"$h" 2>&1
    printf "         exit=%s lines=%s bytes=%s xm2ndp=%s (a whole list is 447 lines)\n" \
        "$?" "$(wc -l <"$h")" "$(wc -c <"$h")" "$(grep -ci xm2ndp "$h")"
    printf "         llc %s\n" "$(sha256sum "$LLC" | cut -c1-16)"
    printf "         %s\n" "$(ldd "$LLC" 2>&1 | grep -i libLLVM | head -1)"
    printf "         %s | libstdc++ %s\n" \
        "$(grep -h '^PRETTY_NAME=' /etc/os-release 2>/dev/null | cut -d'"' -f2)" \
        "$(dpkg-query -W -f='${Version}' libstdc++6 2>/dev/null)"
    printf "         tail: %s\n" "$(tail -1 "$h")"
    rm -f "$h"
}

tier_t1() {
    section "Codegen contract — each workload lowers to the device code we expect"
    if ./scripts/build.sh >/dev/null 2>&1; then pass "build.sh emitted device IR + asm for every benchmark"
    else fail "build.sh emits device code for every benchmark" "run ./scripts/build.sh to see which failed"; return; fi
    if ./scripts/verify.sh >/dev/null 2>&1; then pass "verify.sh — atomics stay one instruction, no frame growth"
    else fail "verify.sh codegen checks" "run ./scripts/verify.sh to see which check failed (device code kept in out/)"; fi
}

# Compile a benchmark's device IR into a controller-runnable image, the same
# pipeline a real task takes (scripts/timing-run.sh): build.sh emits out/<name>.ll,
# llc lowers it at -code-model=medium (medany), and link-m2ndp.sh links it against
# the launcher with -pie so the loader can relocate it into a pool code slot.
# Prints the failing step's error on the first line so a [FAIL] shows why.
_t2_compile() {  # name out.elf
    local name="$1" out="$2" work rc
    local features="+m,+a,+f,+d,+v,+zvl128b,+zfh,+zvfh,+xm2ndp"
    [ -f "out/$name.ll" ] || ./scripts/build.sh "$name" >/dev/null 2>&1 || { echo "build.sh $name failed to emit device IR"; return 1; }
    work="$(mktemp -d)"
    "$LLC" -mtriple=riscv64 -mattr="$features" -code-model=medium -filetype=obj "out/$name.ll" -o "$work/$name.o" 2>"$work/err" &&
    M2NDP_DET="${M2NDP_DET:-$REPO/third_party/m2ndp-detour}" "$REPO/scripts/link-m2ndp.sh" "$work/$name.o" "$out" >"$work/err" 2>&1
    rc=$?; [ "$rc" = 0 ] || head -1 "$work/err"; rm -rf "$work"; return $rc
}

# device_main-only cases: a mojo workload whose device_main computes on the
# controller itself -- its own stack, a non-inlined call, arithmetic -- and
# launches nothing. Run each on the controller and assert the shape of that run:
# zero launches, load/store issues that actually happened, no abort. It is a
# self-contained task with no external buffer, so there is no golden answer to
# check -- the tier verifies the controller path runs and moves data, not a value.
tier_t2() {
    section "Controller / device_main — runs on the controller and moves data"
    local det="${M2NDP_DET:-$REPO/third_party/m2ndp-detour}"
    local runner="$det/build/bin/devmain_run"
    local cfg="$det/config/performance/M2NDP/m2ndp.config"
    [ -x "$runner" ] || { skip "devmain_run not built ($runner)"; return; }
    # devmain_run is a shared-library build with no rpath to the libs shipped
    # beside it (m2ndp_run carries one); point it at both the detour and the LLVM
    # lib dirs so it loads here and in CI.
    local libs="$det/build/lib:$(dirname "$LLC")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    local any=0 name
    for name in $(manifest_names t2); do
        any=1
        local elf err out rc
        elf="$(mktemp --suffix=.elf)"
        if ! err="$(_t2_compile "$name" "$elf" 2>&1)"; then
            fail "$name — compiles into a controller image" "${err%%$'\n'*}"; rm -f "$elf"; continue
        fi
        out="$(LD_LIBRARY_PATH="$libs" M2NDP_STATS=1 "$runner" "$elf" 64 "$cfg" 2>&1)"; rc=$?
        rm -f "$elf"
        if [ "$rc" -ge 128 ]; then fail "$name — device_main runs to completion" "the run aborted (exit $rc); the controller path crashed"; continue; fi
        local launches ldst
        launches="$(printf '%s\n' "$out" | sed -n 's/^STAT launches=//p')"
        ldst="$(printf '%s\n' "$out" | sed -n 's/^STAT ctrl.ldst_issue=//p')"
        [ "${launches:-x}" = 0 ] || { fail "$name — device_main launches nothing" "expected launches=0, got ${launches:-none}"; continue; }
        { [ -n "$ldst" ] && [ "$ldst" -gt 0 ]; } 2>/dev/null || { fail "$name — load/store traffic reaches memory" "expected ctrl.ldst_issue>0, got ${ldst:-none}"; continue; }
        pass "$name — launches=$launches, ld/st=$ldst"
    done
    [ "$any" = 1 ] || skip "no t2 cases in the manifest"
}

tier_t3() {
    section "Basic-op kernels — one-operation kernels on the controller"
    skip "pending op-kernel fixtures + the stat harness"
}

tier_t4() {
    section "Workload end-to-end — a whole workload launches and computes correctly"
    local any=0 w expect note out reason
    for w in $(manifest_names t4); do
        any=1
        expect="$(manifest_expect "$w")"
        note="$(manifest_note "$w")"
        # An xfail case is only excused when we are not requiring all-green.
        [ "$REQUIRE_ALL_GREEN" = 1 ] && expect=pass
        # host-run.sh is the mojo user's own path: build the host program, which
        # fills the pool, launches the task on the simulator, and checks the answer
        # against one it computes itself. "[host] <name> ok" is that self-check.
        out="$(./scripts/host-run.sh "$w" 2>&1)"
        if printf '%s\n' "$out" | grep -q "\[host\] $w ok"; then
            case "$expect" in
                xfail) xpass "$w" ;;
                *)     pass "$w — filled inputs, launched on the simulator, answer matched" ;;
            esac
        else
            reason="$(printf '%s\n' "$out" | grep -oiE 'RENAMING PANIC|no symbol at kinfo address|no kernel at 0x[0-9a-f]+|wrong at [^ ]+ ?: [^ ]+ expected [^ ]+' | head -1)"
            case "$expect" in
                xfail) xfail "$w — ${note:-known-broken}" "${reason:-rerun ./scripts/host-run.sh $w}" ;;
                *)     fail "$w — launch and check on the simulator" "${reason:-rerun ./scripts/host-run.sh $w}" ;;
            esac
        fi
    done
    [ "$any" = 1 ] || skip "no t4 cases in the manifest"
}

# The opcode-coverage map: which RVV/scalar/CSR/M2NDP opcodes each workload
# lowers to, and the featured families the Spike port validated. Not a pass/fail
# gate -- a report -- so it runs on demand (./scripts/test.sh coverage) and after
# a full `all` run, never stopping the suite.
tier_coverage() {
    section "Opcode coverage — the surface each workload exercises"
    if [ -x ./scripts/opcode-map.sh ]; then ./scripts/opcode-map.sh --summary
    else skip "scripts/opcode-map.sh not present"; fi
}

# Long tier names for the "stopped at" line, keyed by the t0..t4 argument.
tier_title() { case "$1" in
    t0) echo "Toolchain" ;; t1) echo "Codegen contract" ;; t2) echo "Controller / device_main" ;;
    t3) echo "Basic-op kernels" ;; t4) echo "Workload end-to-end" ;; coverage) echo "Opcode coverage" ;; *) echo "$1" ;; esac; }

TIERS=("$@")
[ ${#TIERS[@]} -eq 0 ] && TIERS=(all)
# "all" is the tiers cheapest-first, then the coverage report at the end.
[ "${TIERS[0]}" = all ] && TIERS=(t0 t1 t2 t3 t4 coverage)

for t in "${TIERS[@]}"; do
    case "$t" in
        t0) tier_t0 ;; t1) tier_t1 ;; t2) tier_t2 ;; t3) tier_t3 ;; t4) tier_t4 ;; coverage) tier_coverage ;;
        *) echo "unknown tier: $t (want t0..t4, coverage, or all)"; exit 2 ;;
    esac
    if [ "$FAIL" = 1 ]; then
        summary
        printf "${R}[stopped]${N} %s failed — see [FAIL] above\n" "$(tier_title "$t")"
        exit 1
    fi
done

summary
