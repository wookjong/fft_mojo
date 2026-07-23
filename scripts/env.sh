# Environment variables needed to run Mojo.
#   source scripts/env.sh
# If MOJO_ROOT is already exported it is used as-is; otherwise ./toolchain.

if [ -z "${MOJO_ROOT:-}" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
    export MOJO_ROOT="$_here/toolchain"
fi

if [ ! -d "$MOJO_ROOT" ]; then
    echo "MOJO_ROOT($MOJO_ROOT) does not exist. Run ./scripts/setup.sh first." >&2
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

# Put src/ on the module search path (the m2ndp library).
_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export MOJO_PYTHON_LIBRARY="" # unused
export M2NDP_SRC="$_repo/src"
