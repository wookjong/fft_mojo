#!/usr/bin/env bash
# =============================================================================
# Install Mojo (environment setup for the M²NDP PoC).
#
# Downloads three of Modular's nightly wheels and merges them into a single
# directory. After installing, `source scripts/env.sh` sets the environment.
#
# Usage:
#   ./scripts/setup.sh [install_dir]     # default: ./toolchain
#
# NOTE: not every nightly ships a RISC-V backend. If the build later fails
# with "no compiler backend is registered for target 'riscv64-...'", pin a
# known-good build via MOJO_VERSION (see README, "Toolchain requirement").
# =============================================================================
set -euo pipefail

INSTALL_DIR="${1:-$(pwd)/toolchain}"
MOJO_VERSION="${MOJO_VERSION:-}"   # empty means latest nightly

echo "[*] Install location: $INSTALL_DIR"

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }
python3 -c "import pip" 2>/dev/null || { echo "pip required"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[*] Downloading wheels (nightly channel)..."
PKGS="mojo mojo-compiler mojo-compiler-mojo-libs"
if [ -n "$MOJO_VERSION" ]; then
    PKGS="mojo==$MOJO_VERSION mojo-compiler==$MOJO_VERSION mojo-compiler-mojo-libs==$MOJO_VERSION"
fi
# shellcheck disable=SC2086
python3 -m pip download --pre --no-deps -d "$WORK" $PKGS \
    --extra-index-url https://whl.modular.com/nightly/simple/

echo "[*] Merging wheels..."
mkdir -p "$INSTALL_DIR"
for whl in "$WORK"/*.whl; do
    echo "    - $(basename "$whl")"
    # Wheels are plain zips; use python so unzip is not a hard dependency.
    python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
        "$whl" "$WORK/extracted"
done

# The modular/ tree lives either at the archive root or under
# <pkg>.data/platlib/ depending on how the wheel was built. Handle both.
found=0
if [ -d "$WORK/extracted/modular" ]; then
    cp -r "$WORK/extracted/modular/." "$INSTALL_DIR/"
    found=1
fi
for d in "$WORK"/extracted/*.data/platlib/modular; do
    if [ -d "$d" ]; then
        cp -r "$d/." "$INSTALL_DIR/"
        found=1
    fi
done
[ "$found" = 1 ] || { echo "Unrecognized wheel layout"; exit 1; }

# Restore the execute bit before looking for the binary: wheels are zips, and
# python's zipfile drops mode bits on extraction, so everything lands 0644.
chmod +x "$INSTALL_DIR"/bin/* 2>/dev/null || true

# Locate the actual compiler binary.
if [ -x "$INSTALL_DIR/bin/mojo.real" ]; then
    BIN="$INSTALL_DIR/bin/mojo.real"
elif [ -x "$INSTALL_DIR/bin/mojo" ]; then
    BIN="$INSTALL_DIR/bin/mojo"
else
    echo "Install failed: no bin/mojo(.real)"; exit 1
fi

echo ""
echo "[done] $BIN"
"$BIN" --version 2>/dev/null || echo "(version check skipped)"

# Fail early and loudly if this build cannot target RISC-V, rather than
# letting the user discover it as an opaque build.sh failure. Compile a
# throwaway kernel and look for the target-registration error: searching the
# binary for "riscv" strings does not work, since builds that reject the
# target still contain plenty of them.
_probe="$(mktemp -d)"
cat > "$_probe/probe.mojo" <<'PROBE'
from std.gpu.host.compile import _compile_code
def t() -> __mlir_type.`!kgen.target`:
    return __mlir_attr[
        `#kgen.target<triple = "riscv64-unknown-elf", `,
        `arch = "generic-rv64", `,
        `features = "+m,+a,+f,+d,+v,+zvl128b", `,
        `data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
def k(p: UnsafePointer[Float32, MutAnyOrigin]):
    p[0] = p[1] + 1.0
def main():
    comptime x = t()
    print(_compile_code[k, emission_kind="llvm", target=x]().asm)
PROBE
if ! ( cd "$_probe" && MODULAR_MOJO_MAX_PACKAGE_ROOT="$INSTALL_DIR" \
        MODULAR_MOJO_MAX_IMPORT_PATH="$INSTALL_DIR/lib/mojo" \
        MODULAR_HOME="$INSTALL_DIR" LD_LIBRARY_PATH="$INSTALL_DIR/lib" \
        "$BIN" run probe.mojo 2>/dev/null | grep -q riscv64 ); then
    echo ""
    echo "WARNING: this toolchain cannot target RISC-V; ./scripts/build.sh"
    echo "         will fail. Pin a known-good build, e.g."
    echo "         MOJO_VERSION=1.0.0b2.dev2026061203 ./scripts/setup.sh"
fi
rm -rf "$_probe"

echo ""
echo "Next steps:"
echo "  export MOJO_ROOT=$INSTALL_DIR"
echo "  source scripts/env.sh"
echo "  ./scripts/build.sh"
