# Development image for mojo-m2ndp.
#
# Carries the LLVM fork and the M²NDP-Detour timing simulator, so a container can
# compile a benchmark and run it without setting anything up:
#
#   docker run --rm -it ghcr.io/psal-postech/mojo-m2ndp:main
#   ./scripts/build.sh && ./scripts/verify.sh
#
# Neither toolchain is built here. Both arrive prebuilt as release assets that
# image.yml fetches -- LLVM from the fork (tag llvm-<sha>) and Detour from its own
# CI (tag m2ndp-detour-<sha>) -- so the image is an assembly of assets, not a
# build. Nothing of ours is compiled, which is what keeps the image small and the
# push fast. See scripts/build-llvm.sh to build LLVM from source for local work.
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

# APT::Sandbox::User "root": the slurm-ghr builder is rootless podman, where
# apt's drop to the _apt user for downloads cannot setgroups ("Could not switch
# group", exit 100). Fetching as root skips that drop. Inherited by the stages
# below, which are FROM base.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/00no-sandbox

# gcc-riscv64-unknown-elf is for the target, not the host: the simulator
# launchers are compiled with it. It is the largest package here -- about 200 MB
# of newlib multilib -- and it earns it, because without a target compiler the
# image cannot run what it is for. Linking still goes through lld: this binutils
# is older than the LLVM here and rejects the ISA string it emits.
#
# libxml2 and zlib1g are the runtime the prebuilt libLLVM.so links; zstd unpacks
# the LLVM asset (Detour's is a .tar.gz). No conan, cmake, flex or bison: nothing
# of ours is compiled here any more, so the tools that built Detour are gone.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gcc-riscv64-unknown-elf \
        git \
        libxml2 \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
        zlib1g \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Mojo, pinned to the nightly the repo targets. Pinned rather than latest because
# the RISC-V backend was dropped partway through the b3 series -- see
# docs/STATUS.md for which nightlies work. setup.sh warns if the version it
# installed cannot target RISC-V.
ARG MOJO_VERSION=1.0.0b2.dev2026061203
ENV MOJO_ROOT=/opt/mojo
COPY scripts/setup.sh /tmp/setup.sh
RUN MOJO_VERSION="$MOJO_VERSION" /tmp/setup.sh "$MOJO_ROOT" \
    && rm -f /tmp/setup.sh \
    && rm -rf /root/.cache/pip

#===----------------------------------------------------------------------===#
# Assemble: unpack the prebuilt assets (nothing is compiled)
#===----------------------------------------------------------------------===#

FROM base AS assemble

WORKDIR /work

# The two assets image.yml fetched: LLVM as a relocatable install prefix, Detour
# as its harnesses + libNDPSim_lib.so + configs + launch-ABI headers.
COPY llvm-asset /work/llvm-asset
COPY detour-asset /work/detour-asset

# --build-arg BUILD_TOOLCHAINS=0 skips this, for a quick check that the Dockerfile
# is sound. The final stage copies from here unconditionally, so the skip branch
# still leaves the directories behind; an image built that way has no toolchain.
ARG BUILD_TOOLCHAINS=1
RUN if [ "$BUILD_TOOLCHAINS" = "1" ]; then \
        mkdir -p build/llvm third_party/m2ndp-detour \
        && tar -C build/llvm --zstd -xf llvm-asset/llvm.tar.zst \
        && tar -C third_party/m2ndp-detour -xzf detour-asset/detour.tar.gz; \
    else \
        mkdir -p build/llvm/bin build/llvm/lib \
                 third_party/m2ndp-detour/build/bin \
                 third_party/m2ndp-detour/build/lib \
                 third_party/m2ndp-detour/config third_party/m2ndp-detour/src; \
        : > build/llvm/lib/libLLVM.so; \
    fi

#===----------------------------------------------------------------------===#
# Image: the assets, and the source that is not an asset
#===----------------------------------------------------------------------===#

FROM base

WORKDIR /work

# LLVM: the tools the scripts run (llc, ld.lld, ...) and the shared library the
# Detour harnesses link. No headers or CMake package -- nothing here builds
# against LLVM, so shipping them was only image weight and upload.
COPY --from=assemble /work/build/llvm/bin /work/build/llvm/bin
COPY --from=assemble /work/build/llvm/lib/libLLVM*.so* /work/build/llvm/lib/

# Detour: the controller harnesses, the shared library they link, the sim configs
# they read, and src/ for the launch-ABI header our launcher includes. The
# launcher and link script are ours (sim/, scripts/), copied with the rest below.
COPY --from=assemble /work/third_party/m2ndp-detour/build/bin /work/third_party/m2ndp-detour/build/bin
COPY --from=assemble /work/third_party/m2ndp-detour/build/lib /work/third_party/m2ndp-detour/build/lib
COPY --from=assemble /work/third_party/m2ndp-detour/config /work/third_party/m2ndp-detour/config
COPY --from=assemble /work/third_party/m2ndp-detour/src /work/third_party/m2ndp-detour/src

# The repository, minus the submodules. Their trees are large and nothing in the
# documented workflows reads them -- the toolchains are already built. Rebuilding
# LLVM or running its lit suite needs a checkout; see docs/SIMULATION.md.
COPY benchmarks /work/benchmarks
COPY docs /work/docs
COPY scripts /work/scripts
COPY sim /work/sim
COPY src /work/src
COPY README.md CLAUDE-m2ndp.md LICENSE /work/

# The harnesses' RUNPATH is a build-machine absolute path, so it does not resolve
# here; LD_LIBRARY_PATH is searched first and carries both shared libraries --
# libNDPSim_lib.so and the libLLVM.so it links.
ENV PATH=/work/build/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/work/build/llvm/lib:/work/third_party/m2ndp-detour/build/lib

CMD ["/bin/bash"]
