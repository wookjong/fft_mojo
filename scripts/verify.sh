#!/usr/bin/env bash
# =============================================================================
# 생성된 산출물이 기대대로인지 검증한다. build.sh 이후 실행.
#   ./scripts/verify.sh
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FAIL=0
check() {
    local desc="$1" cmd="$2" expect="$3"
    local got
    got="$(eval "$cmd" 2>/dev/null)"
    if [ "$got" = "$expect" ]; then
        printf "  OK   %-44s (%s)\n" "$desc" "$got"
    else
        printf "  FAIL %-44s expected=%s got=%s\n" "$desc" "$expect" "$got"
        FAIL=1
    fi
}

echo "[verify] 산출물 검증"
[ -d out ] || { echo "out/ 없음. ./scripts/build.sh 먼저 실행"; exit 1; }

check "RISC-V 타깃" \
      "grep -h 'target triple' out/*.ll | sort -u | grep -c riscv64" "1"
check "M2NDP 심볼 4종" \
      "grep -ho '@__m2ndp_[a-z_]*' out/*.ll | sort -u | wc -l | tr -d ' '" "4"
check "스크래치패드 addrspace(3)" \
      "test \$(grep -c 'addrspace(3)' out/spmv.ll) -ge 4 && echo yes" "yes"
check "RVV vsetivli" \
      "grep -c 'vsetivli' out/vadd_simd.s | tr -d ' '" "1"
check "RVV 벡터 load/add/store" \
      "test \$(grep -cE 'vle32\.v|vfadd\.vv|vse32\.v' out/vadd_simd.s) -ge 3 && echo yes" "yes"
check "그룹 배리어 호출" \
      "grep -q 'call void @__m2ndp_barrier' out/spmv.ll && echo yes" "yes"

echo ""
if [ "$FAIL" = 0 ]; then
    echo "[verify] 전부 통과"
else
    echo "[verify] 실패 항목 있음"
    exit 1
fi
