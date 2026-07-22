#!/usr/bin/env bash
# =============================================================================
# 워크로드를 컴파일해 LLVM IR / 어셈블리를 out/ 에 생성한다.
#
#   ./scripts/build.sh              # 전체 워크로드, llvm + asm
#   ./scripts/build.sh vadd         # 특정 워크로드만
#   EMISSION=asm ./scripts/build.sh # 특정 emission만 (llvm|asm)
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# shellcheck source=/dev/null
. "$REPO/scripts/env.sh"

if [ ! -x "$MOJO_BIN" ]; then
    echo "Mojo가 없습니다. ./scripts/setup.sh 를 먼저 실행하세요." >&2
    exit 1
fi

mkdir -p out
TARGETS="${1:-}"
if [ -z "$TARGETS" ]; then
    TARGETS="$(ls workloads/*.mojo | xargs -n1 basename | sed 's/\.mojo$//')"
fi
EMISSIONS="${EMISSION:-llvm asm}"

# src/ 를 import 경로에 추가하기 위해 workloads 를 src 와 함께 임시 디렉토리에 둔다
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp src/*.mojo "$STAGE/"

for name in $TARGETS; do
    wl="workloads/$name.mojo"
    [ -f "$wl" ] || { echo "!! $wl 없음"; continue; }
    for em in $EMISSIONS; do
        # emission_kind 치환본을 스테이지에 생성
        sed "s/emission_kind=\"llvm\"/emission_kind=\"$em\"/" "$wl" > "$STAGE/$name.mojo"
        ext="ll"; [ "$em" = "asm" ] && ext="s"
        outfile="out/${name}.${ext}"
        printf "[build] %-12s %-4s -> %s\n" "$name" "$em" "$outfile"
        ( cd "$STAGE" && timeout 180 "$MOJO_BIN" run "$name.mojo" ) > "$outfile" 2>"$STAGE/err.txt"
        if [ ! -s "$outfile" ]; then
            echo "  !! 실패:"; sed 's/^/     /' "$STAGE/err.txt" | head -5
            rm -f "$outfile"
        fi
    done
done

echo ""
echo "생성된 파일:"
ls -la out/ 2>/dev/null | tail -n +2
