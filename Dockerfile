# Development image for mojo-m2ndp.
#
# Carries a built LLVM fork and a built Spike, so a container can compile a
# benchmark and run it without setting anything up:
#
#   docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
#   ./scripts/build.sh && ./scripts/verify.sh
#   ./scripts/spike-smoke.sh
#
# Both submodules have to be in the build context -- the LLVM one is private,
# so the image cannot clone it without carrying a token in a layer:
#
#   git submodule update --init --depth 1 --recursive
#   docker build -t ghcr.io/psal-postech/mojo-m2ndp:dev .
#
# Do not mount over /work: the toolchains live there and a mount hides them.
# Mount a checkout somewhere else and point the scripts at it.
#
# Built and published by .github/workflows/image.yml.

#===----------------------------------------------------------------------===#
# Base: the packages the image needs
#===----------------------------------------------------------------------===#

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# device-tree-compiler is Spike's: its configure hard-errors without dtc, and
# says so without saying which package to install.
#
# gcc-riscv64-unknown-elf is for the target, not the host: the simulator
# launchers are compiled with it. It is the largest single package here --
# about 200 MB, most of it newlib multilib variants -- and it pulls in the
# matching binutils for another 26 MB. Worth it, because without a target
# compiler the image cannot run what it is for. Note that linking still goes
# through lld: this binutils is older than the LLVM in the image and rejects
# the ISA string it emits.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        device-tree-compiler \
        gcc-riscv64-unknown-elf \
        git \
        libxml2-dev \
        ninja-build \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Mojo, pinned to the nightly the repo targets. Pinned rather than latest
# because the RISC-V backend was dropped partway through the b3 series — see
# docs/STATUS.md for which nightlies work. setup.sh warns if the version it
# installed cannot target RISC-V.
ARG MOJO_VERSION=1.0.0b2.dev2026061203
ENV MOJO_ROOT=/opt/mojo
COPY scripts/setup.sh /tmp/setup.sh
RUN MOJO_VERSION="$MOJO_VERSION" /tmp/setup.sh "$MOJO_ROOT" \
    && rm -f /tmp/setup.sh \
    && rm -rf /root/.cache/pip

#===----------------------------------------------------------------------===#
# Builder: everything that is thrown away afterwards
#===----------------------------------------------------------------------===#

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends ccache \
    && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache \
    PATH=/usr/lib/ccache:$PATH
RUN mkdir -p /ccache && chmod 777 /ccache

WORKDIR /work

# The submodules and the build scripts first, on their own layer. Everything
# after this is the 25-minute part, and copying the whole tree up front would
# throw it away for a change to a benchmark or a document.
COPY third_party /work/third_party
COPY scripts /work/scripts
COPY sim /work/sim

# --build-arg BUILD_TOOLCHAINS=0 skips this, for a quick check that the
# Dockerfile itself is sound without waiting for LLVM. The final stage copies
# from here unconditionally, so the skip branch still has to leave the
# directories behind; an image built that way has no compiler in it.
ARG BUILD_TOOLCHAINS=1
RUN if [ "$BUILD_TOOLCHAINS" = "1" ]; then \
        test -f third_party/llvm-project/llvm/CMakeLists.txt \
          || { echo "LLVM submodule missing from the build context" >&2; exit 1; }; \
        test -f third_party/riscv-isa-sim/configure \
          || { echo "riscv-isa-sim submodule missing from the build context" >&2; exit 1; }; \
        ./scripts/build-llvm.sh \
        && ./scripts/build-spike.sh; \
    else \
        mkdir -p build/llvm/bin build/spike/install/bin build/spike/install/lib; \
        : > build/spike/libm2ndp_ext.so; \
    fi

# Debug symbols are the whole difference between an image you can pull and one
# you cannot: Spike's install alone is 1.8 GB with them and 70 MB without.
# Nothing here is debugged with a symbol table -- llvm-objdump and the lit
# suite work on the target's output, not on the tools.
RUN find build -type f \( -name 'spike*' -o -name '*.so' -o -name '*.so.*' \) \
        -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find build/llvm/bin build/spike/install/bin -type f -exec strip {} + \
        2>/dev/null || true

#===----------------------------------------------------------------------===#
# Image: the artifacts, and the source that is not an artifact
#===----------------------------------------------------------------------===#

FROM base

WORKDIR /work

# The built tools, at the paths the scripts look for them at.
COPY --from=builder /work/build/llvm/bin /work/build/llvm/bin
COPY --from=builder /work/build/spike/install /work/build/spike/install
COPY --from=builder /work/build/spike/libm2ndp_ext.so /work/build/spike/

# Spike's headers, which the M²NDP extension is compiled against. 11 MB, and
# without them the extension cannot be rebuilt in the container.
COPY --from=builder /work/third_party/riscv-isa-sim /work/third_party/riscv-isa-sim

# The repository, minus the LLVM submodule. That tree is 2.6 GB and nothing in
# the documented workflows reads it -- the compiler is already built. Rebuilding
# LLVM or running its lit suite needs a checkout; see docs/SIMULATION.md.
COPY benchmarks /work/benchmarks
COPY docs /work/docs
COPY scripts /work/scripts
COPY sim /work/sim
COPY src /work/src
COPY README.md CLAUDE-m2ndp.md LICENSE /work/

ENV PATH=/work/build/spike/install/bin:/work/build/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/work/build/spike/install/lib

CMD ["/bin/bash"]
