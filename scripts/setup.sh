#!/usr/bin/env bash
# =============================================================================
# Mojo 설치 (M²NDP PoC 환경 셋업)
#
# Modular의 nightly wheel 3개를 받아 하나의 디렉토리로 병합한다.
# 설치 후 `source scripts/env.sh` 로 환경변수를 잡으면 된다.
#
# 사용법:
#   ./scripts/setup.sh [설치경로]      # 기본: ./toolchain
# =============================================================================
set -euo pipefail

INSTALL_DIR="${1:-$(pwd)/toolchain}"
MOJO_VERSION="${MOJO_VERSION:-}"   # 비우면 최신 nightly

echo "[*] 설치 위치: $INSTALL_DIR"

command -v python3 >/dev/null || { echo "python3 필요"; exit 1; }
python3 -c "import pip" 2>/dev/null || { echo "pip 필요"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[*] wheel 다운로드 중 (nightly 채널)..."
PKGS="mojo mojo-compiler mojo-compiler-mojo-libs"
if [ -n "$MOJO_VERSION" ]; then
    PKGS="mojo==$MOJO_VERSION mojo-compiler==$MOJO_VERSION mojo-compiler-mojo-libs==$MOJO_VERSION"
fi
# shellcheck disable=SC2086
python3 -m pip download --pre --no-deps -d "$WORK" $PKGS \
    --extra-index-url https://whl.modular.com/nightly/simple/

echo "[*] wheel 병합 중..."
mkdir -p "$INSTALL_DIR"
for whl in "$WORK"/*.whl; do
    echo "    - $(basename "$whl")"
    unzip -q -o "$whl" -d "$WORK/extracted"
done

# wheel 안의 modular/ 디렉토리를 설치 경로로 병합
if [ -d "$WORK/extracted/modular" ]; then
    cp -r "$WORK/extracted/modular/." "$INSTALL_DIR/"
else
    # 레이아웃이 다르면 bin/ 을 가진 디렉토리를 찾아 병합
    found=0
    for d in "$WORK"/extracted/*/; do
        if [ -d "$d/bin" ]; then
            cp -r "$d/." "$INSTALL_DIR/"
            found=1
        fi
    done
    [ "$found" = 1 ] || { echo "wheel 레이아웃을 알 수 없음"; exit 1; }
fi

# 실제 컴파일러 바이너리 확인
if [ -x "$INSTALL_DIR/bin/mojo.real" ]; then
    BIN="$INSTALL_DIR/bin/mojo.real"
elif [ -x "$INSTALL_DIR/bin/mojo" ]; then
    BIN="$INSTALL_DIR/bin/mojo"
else
    echo "설치 실패: bin/mojo(.real) 없음"; exit 1
fi
chmod +x "$INSTALL_DIR"/bin/* 2>/dev/null || true

echo ""
echo "[완료] $BIN"
"$BIN" --version 2>/dev/null || echo "(버전 확인 생략)"
echo ""
echo "다음 단계:"
echo "  export MOJO_ROOT=$INSTALL_DIR"
echo "  source scripts/env.sh"
echo "  ./scripts/build.sh"
