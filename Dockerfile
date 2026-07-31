# Development image for mojo-m2ndp.
#
# Carries a built LLVM fork and the M²NDP-Detour timing simulator, so a
# container can compile a benchmark and run it without setting anything up:
#
#   docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
#   ./scripts/build.sh && ./scripts/verify.sh
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

# gcc-riscv64-unknown-elf is for the target, not the host: the simulator
# launchers are compiled with it. It is the largest single package here --
# about 200 MB, most of it newlib multilib variants -- and it pulls in the
# matching binutils for another 26 MB. Worth it, because without a target
# compiler the image cannot run what it is for. Note that linking still goes
# through lld: this binutils is older than the LLVM in the image and rejects
# the ISA string it emits.
#
# flex and bison generate M²NDP-Detour's config/trace grammar; its CMake calls
# them at configure time, so the timing build cannot start without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bison \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flex \
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

# conan fetches M²NDP-Detour's C++ dependencies. Pinned to the 1.x line: its
# CMake uses conan_basic_setup()/conanbuildinfo.cmake, which conan 2 dropped.
# The default profile links libstdc++11 (the C++11 string ABI): Detour and the
# LLVM its decoder links against both build with _GLIBCXX_USE_CXX11_ABI=1, so
# the dependencies must match or std::string symbols do not resolve at link time.
RUN python3 -m pip install --no-cache-dir "conan==1.56.0" \
    && rm -rf /root/.cache/pip \
    && conan profile new default --detect \
    && conan profile update settings.compiler.libcxx=libstdc++11 default

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

RUN apt-get update && apt-get install -y --no-install-recommends ccache zstd \
    && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache \
    PATH=/usr/lib/ccache:$PATH
RUN mkdir -p /ccache && chmod 777 /ccache

WORKDIR /work

# The detour submodule and the build scripts first, on their own layer. LLVM is
# not built here: it arrives prebuilt in llvm-asset/, a relocatable install
# prefix that image.yml fetches from the fork's release (tag llvm-<sha>) instead
# of cloning and building the ~GB fork. See scripts/build-llvm.sh to build it.
COPY third_party /work/third_party
COPY scripts /work/scripts
COPY sim /work/sim
COPY llvm-asset /work/llvm-asset

# --build-arg BUILD_TOOLCHAINS=0 skips this, for a quick check that the
# Dockerfile itself is sound. The final stage copies from here unconditionally,
# so the skip branch still has to leave the directories behind; an image built
# that way has no compiler in it.
ARG BUILD_TOOLCHAINS=1
RUN if [ "$BUILD_TOOLCHAINS" = "1" ]; then \
        mkdir -p build/llvm \
        && tar -C build/llvm --zstd -xf llvm-asset/llvm.tar.zst \
        && cmake -S third_party/m2ndp-detour -B third_party/m2ndp-detour/build \
             -DPERFORMANCE_BUILD=1 \
        && cmake --build third_party/m2ndp-detour/build -j"$(nproc)" \
             --target m2ndp_run dev_launch dev_launch_loop dev_launch_masked; \
    else \
        mkdir -p build/llvm/bin build/llvm/lib/cmake build/llvm/include \
                 third_party/m2ndp-detour/build/bin; \
        : > build/llvm/lib/libLLVM.so; \
    fi

# Strip M²NDP-Detour's harnesses. The LLVM tools arrive stripped in the asset.
RUN find third_party/m2ndp-detour/build/bin -type f -exec strip {} + 2>/dev/null || true

#===----------------------------------------------------------------------===#
# Image: the artifacts, and the source that is not an artifact
#===----------------------------------------------------------------------===#

FROM base

WORKDIR /work

# The built tools, at the paths the scripts look for them at.
COPY --from=builder /work/build/llvm/bin /work/build/llvm/bin

# LLVM as a shared library, plus its headers and CMake package: the timing
# harnesses above link libLLVM.so, and M²NDP-Detour's own CI rebuilds against it
# without carrying 635 MB of static archives.
COPY --from=builder /work/build/llvm/lib/libLLVM*.so* /work/build/llvm/lib/
COPY --from=builder /work/build/llvm/lib/cmake /work/build/llvm/lib/cmake
COPY --from=builder /work/build/llvm/include /work/build/llvm/include

# M²NDP-Detour's controller harnesses (built above), the sim config they read,
# and src/ for the launch ABI header our launcher includes. The launcher and the
# link script are ours (sim/, scripts/), copied with the rest of the tree.
COPY --from=builder /work/third_party/m2ndp-detour/build/bin /work/third_party/m2ndp-detour/build/bin
COPY --from=builder /work/third_party/m2ndp-detour/config /work/third_party/m2ndp-detour/config
COPY --from=builder /work/third_party/m2ndp-detour/src /work/third_party/m2ndp-detour/src

# The repository, minus the LLVM submodule. That tree is 2.6 GB and nothing in
# the documented workflows reads it -- the compiler is already built. Rebuilding
# LLVM or running its lit suite needs a checkout; see docs/SIMULATION.md.
COPY benchmarks /work/benchmarks
COPY docs /work/docs
COPY scripts /work/scripts
COPY sim /work/sim
COPY src /work/src
COPY README.md CLAUDE-m2ndp.md LICENSE /work/

ENV PATH=/work/build/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/work/build/llvm/lib

CMD ["/bin/bash"]
