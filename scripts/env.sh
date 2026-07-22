# Mojo 실행에 필요한 환경변수.
#   source scripts/env.sh
# MOJO_ROOT를 미리 export 해두면 그것을 쓰고, 없으면 ./toolchain 을 쓴다.

if [ -z "${MOJO_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
    export MOJO_ROOT="$_here/toolchain"
fi

if [ ! -d "$MOJO_ROOT" ]; then
    echo "MOJO_ROOT($MOJO_ROOT)이 없습니다. ./scripts/setup.sh 를 먼저 실행하세요." >&2
fi

export LD_LIBRARY_PATH="$MOJO_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODULAR_MOJO_MAX_PACKAGE_ROOT="$MOJO_ROOT"
export MODULAR_MOJO_MAX_IMPORT_PATH="$MOJO_ROOT/lib/mojo"
export MODULAR_HOME="$MOJO_ROOT"

if [ -x "$MOJO_ROOT/bin/mojo.real" ]; then
    export MOJO_BIN="$MOJO_ROOT/bin/mojo.real"
else
    export MOJO_BIN="$MOJO_ROOT/bin/mojo"
fi
export MODULAR_MOJO_MAX_DRIVER_PATH="$MOJO_BIN"

# src/ 를 모듈 검색 경로에 추가 (m2ndp 라이브러리)
_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export MOJO_PYTHON_LIBRARY="" # 미사용
export M2NDP_SRC="$_repo/src"
