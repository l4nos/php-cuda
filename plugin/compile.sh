#!/usr/bin/env bash
#
# compile.sh: build and install the PHP CUDA extension.
#
# Usage:
#   ./compile.sh [--test] [--uninstall]
#
# Environment overrides:
#   CUDA_PATH       CUDA toolkit root (default: autodetect)
#   CUDNN_PATH      cuDNN root      (default: same as CUDA; "no" disables)
#   CUDA_ARCH       GPU arch list, e.g. "75;80;90" or "native" (default: native)
#   ENABLE_OPENMP=1 Build CPU fallback with OpenMP
#   PHP_CONFIG      Path to php-config (default: autodetect)

set -euo pipefail

cd "$(dirname "$0")"

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Locate PHP development files
# ---------------------------------------------------------------------------
find_php_config() {
    if [ -n "${PHP_CONFIG:-}" ] && [ -x "$PHP_CONFIG" ]; then
        echo "$PHP_CONFIG"; return 0
    fi
    for candidate in php-config /usr/local/bin/php-config /usr/bin/php-config; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}

PHP_CONFIG_BIN="$(find_php_config)" || die "php-config not found. Install the PHP development package (e.g. apt install php-dev)."
PHP_VERSION="$("$PHP_CONFIG_BIN" --version)"
PHP_EXTENSION_DIR="$("$PHP_CONFIG_BIN" --extension-dir)"
log "PHP $PHP_VERSION (extension dir: $PHP_EXTENSION_DIR)"

PHP_VERSION_MAJOR="${PHP_VERSION%%.*}"
PHP_VERSION_MINOR="$(echo "$PHP_VERSION" | cut -d. -f2)"
if [ "$PHP_VERSION_MAJOR" -lt 8 ] || { [ "$PHP_VERSION_MAJOR" -eq 8 ] && [ "$PHP_VERSION_MINOR" -lt 1 ]; }; then
    die "PHP 8.1 or higher is required (found $PHP_VERSION)"
fi

command -v phpize >/dev/null 2>&1 || die "phpize not found. Install the PHP development package."

# ---------------------------------------------------------------------------
# Locate the CUDA toolkit
# ---------------------------------------------------------------------------
find_cuda() {
    if [ -n "${CUDA_PATH:-}" ] && [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
        echo "$CUDA_PATH"; return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        (cd "$(dirname "$(command -v nvcc)")/.." && pwd); return 0
    fi
    for path in /usr/local/cuda /usr/local/cuda-* /opt/cuda /usr/lib/cuda; do
        if [ -f "$path/include/cuda_runtime.h" ]; then
            echo "$path"; return 0
        fi
    done
    return 1
}

CUDA_DIR="$(find_cuda)" || die "CUDA toolkit not found. Install cuda-toolkit (11.8+) or set CUDA_PATH."
log "CUDA toolkit: $CUDA_DIR"

NVCC="$CUDA_DIR/bin/nvcc"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc)" || die "nvcc not found"

NVCC_VERSION="$("$NVCC" --version | awk '/release/ {print $5}' | tr -d ',')"
log "nvcc version: $NVCC_VERSION"

# Mixed-toolchain detection: toolkit version.json vs nvcc
if [ -f "$CUDA_DIR/version.json" ]; then
    TOOLKIT_VERSION="$(awk -F'"' '/"cuda"/{getline; getline; print $4; exit}' "$CUDA_DIR/version.json" 2>/dev/null || true)"
    if [ -n "$TOOLKIT_VERSION" ] && [ "${TOOLKIT_VERSION%%.*}" != "${NVCC_VERSION%%.*}" ]; then
        die "Mixed CUDA toolchain detected: nvcc is $NVCC_VERSION but $CUDA_DIR is toolkit $TOOLKIT_VERSION. Fix PATH/CUDA_PATH so they match."
    fi
fi

# ---------------------------------------------------------------------------
# Driver / runtime compatibility check
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print $1}' | head -n1 || true)"
    if [ -n "$DRIVER_CUDA" ]; then
        log "Driver supports CUDA runtime up to: $DRIVER_CUDA"
        DRIVER_MAJOR="${DRIVER_CUDA%%.*}"
        DRIVER_MINOR="$(echo "$DRIVER_CUDA" | cut -d. -f2)"
        NVCC_MAJOR="${NVCC_VERSION%%.*}"
        NVCC_MINOR="$(echo "$NVCC_VERSION" | cut -d. -f2)"
        if [ "$NVCC_MAJOR" -gt "$DRIVER_MAJOR" ] || { [ "$NVCC_MAJOR" -eq "$DRIVER_MAJOR" ] && [ "$NVCC_MINOR" -gt "$DRIVER_MINOR" ]; }; then
            warn "Toolkit runtime ($NVCC_VERSION) is newer than the driver's supported runtime ($DRIVER_CUDA)."
            warn "Binaries may fail to load. Upgrade the NVIDIA driver or install the cuda-compat package."
        fi
    fi
else
    warn "nvidia-smi not found; skipping driver/runtime compatibility check (build-only host?)"
fi

# ---------------------------------------------------------------------------
# Optional components
# ---------------------------------------------------------------------------
CONFIGURE_ARGS="--with-cuda=$CUDA_DIR"

if [ "${CUDNN_PATH:-yes}" = "no" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-cudnn=no"
elif [ -n "${CUDNN_PATH:-}" ]; then
    [ -f "$CUDNN_PATH/include/cudnn.h" ] || die "cudnn.h not found under CUDNN_PATH=$CUDNN_PATH"
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-cudnn=$CUDNN_PATH"
fi

if [ -n "${CUDA_ARCH:-}" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-cuda-arch=$CUDA_ARCH"
fi

if [ "${ENABLE_NVTX:-0}" = "1" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --with-nvtx"
fi

if [ "${ENABLE_OPENMP:-0}" = "1" ]; then
    CONFIGURE_ARGS="$CONFIGURE_ARGS --enable-openmp"
fi

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--uninstall" ]; then
    PHP_INI_DIR="$("$PHP_CONFIG_BIN" --ini-dir)"
    log "Removing extension and configuration"
    sudo rm -f "$PHP_EXTENSION_DIR/cuda.so" "$PHP_INI_DIR/cuda.ini"
    log "Uninstalled. Restart your PHP processes."
    exit 0
fi

# ---------------------------------------------------------------------------
# Clean previous build artifacts (never config.m4 / config.w32 / sources)
# ---------------------------------------------------------------------------
log "Cleaning previous build artifacts"
rm -rf .libs modules build autom4te.cache libtool
rm -f  *.lo *.la *.o cuda.la
rm -f  configure configure.in aclocal.m4
rm -f  config.h config.h.in config.h.in~ config.log config.status config.nice
rm -f  Makefile Makefile.objects Makefile.fragments Makefile.global
rm -f  run-tests.php

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
log "Running phpize"
phpize

log "Configuring: ./configure $CONFIGURE_ARGS"
# shellcheck disable=SC2086
./configure $CONFIGURE_ARGS

log "Building"
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

[ -f "modules/cuda.so" ] || die "Build failed: modules/cuda.so not found"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
log "Installing"
sudo make install

PHP_INI_DIR="$("$PHP_CONFIG_BIN" --ini-dir)"
if [ ! -f "$PHP_INI_DIR/cuda.ini" ]; then
    log "Writing $PHP_INI_DIR/cuda.ini"
    echo "extension=cuda.so" | sudo tee "$PHP_INI_DIR/cuda.ini" >/dev/null
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
if php -m | grep -q '^cuda$'; then
    log "CUDA extension installed and enabled"
    php -r 'printf("Devices visible: %d\n", cuda_device_count());' || true
else
    warn "Extension installed but not loaded by the CLI SAPI. Check $PHP_INI_DIR/cuda.ini."
fi

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--test" ]; then
    log "Running test suite"
    php run-tests.php -q -x -d extension="$PWD/modules/cuda.so" ../tests/ || true
fi

log "Done. Restart long-running PHP processes (php-fpm, RoadRunner, etc.) to pick up the new build."
