#!/usr/bin/env bash
# =============================================================================
# The opcode-coverage map: for every workload, the distinct RVV / scalar / CSR /
# M2NDP opcodes its device code lowers to. The device .s is the source of truth
# -- the same assembly a launch feeds the loader (scripts/build.sh emits it with
# our llc) -- so this reports what the suite actually exercises, not a wish list.
#
#   ./scripts/opcode-map.sh              # per-workload map + featured cross-ref
#   ./scripts/opcode-map.sh --md         # the same, as a Markdown table for docs
#   ./scripts/opcode-map.sh --summary    # one compact line per workload
#
# Missing out/*.s are built first (scripts/build.sh). A workload with no .s is
# reported as un-lowered rather than skipped, so a codegen regression is visible.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODE="${1:-full}"

# Featured opcode families -- the surface the Spike executor port validated.
# "label<TAB>regex"; regex matches the mnemonic at the start of an asm line.
FEATURED='vec->scalar move (vfmv.f.s/vmv.x.s)	vfmv\.f\.s|vmv\.x\.s
scalar->vec broadcast (vfmv.v.f/vmv.v.x)	vfmv\.v\.f|vmv\.v\.f|vmv\.v\.x
int<->fp convert (vfcvt.x.f.v/f.x.v)	vfcvt\.[xf]\.[fx]\.v
fp narrow/widen (vfncvt/vfwcvt)	vfncvt|vfwcvt
slide (vslidedown/vslideup)	vslidedown|vslideup
FMA family (vfmacc/vfmadd/vfmsac)	vfn?macc|vfn?madd|vfn?msac|vfn?msub
reduction (vfredmax/vredmin/...)	v[f]?red[a-z]*\.vs
CSR frm (csrrwi frm, fsrm/fsrmi)	fsrm|fsrmi|csrrwi?|frrm
M2NDP vector AMO (int/fp)	m2ndp\.v[f]?amoadd
M2NDP scalar-fp AMO (famoadd/famomax)	m2ndp\.famo
M2NDP kernel exit	m2ndp\.exit
mask/predicate (vmslt/vmfgt/vmand.mm/...)	vms[a-z]*\.|vmf[a-z]*\.|vmand\.mm|vmor\.mm|vfirst\.m|vmerge
gather (vrgather)	vrgather
vector fp arithmetic (vfadd/vfmul/...)	vf(add|sub|mul|div|max|min|abs|sgnj|w)
vector int arithmetic (vadd/vand/vsll/...)	v(add|sub|and|or|xor|sll|srl|sra|id)\.
vector load/store (vle/vse)	vle[0-9]+\.v|vse[0-9]+\.v
scalar fp (fadd.s/fsqrt.s/fmadd.s/...)	f(add|sub|mul|div|sqrt|madd|nmsub|cvt)\.'

# Distinct mnemonics in a workload's .s, one per line, pseudo-ops and all.
_mnemonics() {  # file
    grep -oE '^[[:space:]]+[a-z][a-zA-Z0-9._]+' "$1" 2>/dev/null \
        | sed 's/^[[:space:]]*//' | grep -vE '^\.' | sort -u
}

workloads() { ls out/*.s 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.s$//'; }

# Build any missing device code once up front.
if ! ls out/*.s >/dev/null 2>&1; then
    echo "[opcode-map] no out/*.s yet -- running build.sh" >&2
    ./scripts/build.sh >/dev/null 2>&1 || { echo "build.sh failed; run it directly" >&2; exit 1; }
fi

if [ "$MODE" = --summary ]; then
    for w in $(workloads); do
        ops="$(_mnemonics "out/$w.s")"
        n="$(printf '%s\n' "$ops" | grep -c .)"
        vec="$(printf '%s\n' "$ops" | grep -cE '^v')"
        printf "  %-20s %2d opcodes (%d vector)  %s\n" "$w" "$n" "$vec" \
            "$(printf '%s\n' "$ops" | grep -E '^m2ndp' | tr '\n' ' ')"
    done
    exit 0
fi

# The featured cross-reference: each family -> the workloads that lower to it.
if [ "$MODE" = --md ]; then
    echo "| Opcode family (port-validated) | Workloads exercising it |"
    echo "|---|---|"
    while IFS=$'\t' read -r label rx; do
        [ -n "$label" ] || continue
        hits=""
        for w in $(workloads); do
            _mnemonics "out/$w.s" | grep -qE "^($rx)" && hits="$hits $w"
        done
        printf "| %s | %s |\n" "$label" "$(echo $hits | sed 's/ /, /g')"
    done <<< "$FEATURED"
    exit 0
fi

# Default: human-readable report.
if [ -t 1 ]; then B='\033[1m'; D='\033[2m'; N='\033[0m'; else B=; D=; N=; fi

printf "${B}Opcode coverage -- distinct mnemonics per workload${N}\n"
printf "${D}(from out/*.s, our llc's lowering; %s workloads)${N}\n\n" "$(workloads | grep -c .)"
for w in $(workloads); do
    ops="$(_mnemonics "out/$w.s")"
    n="$(printf '%s\n' "$ops" | grep -c .)"
    printf "  ${B}%-20s${N} %2d opcodes\n" "$w" "$n"
    printf '%s\n' "$ops" | grep -E '^(v|m2ndp|fsrm|csrr)' | tr '\n' ' ' | fmt -w 76 | sed 's/^/      /'
    echo
done

printf "\n${B}Featured opcode families -- the surface the Spike port validated${N}\n\n"
while IFS=$'\t' read -r label rx; do
    [ -n "$label" ] || continue
    hits=""
    for w in $(workloads); do
        _mnemonics "out/$w.s" | grep -qE "^($rx)" && hits="$hits $w"
    done
    printf "  ${B}%-42s${N} %s\n" "$label" "$(echo $hits | sed 's/ /, /g')"
done <<< "$FEATURED"
