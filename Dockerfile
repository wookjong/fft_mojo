# Development image for mojo-m2ndp.
#
# Everything needed to build the LLVM submodule and run the Mojo benchmarks.
#
#   docker build -t ghcr.io/psal-postech/mojo-m2ndp:dev .
#   docker run --rm -it -v "$PWD:/work" ghcr.io/psal-postech/mojo-m2ndp:dev
#
# Built and published by .github/workflows/image.yml.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
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

# ccache keeps LLVM rebuilds tolerable across container restarts when the
# cache directory is mounted from the host.
ENV CCACHE_DIR=/ccache \
    PATH=/usr/lib/ccache:$PATH
RUN mkdir -p /ccache && chmod 777 /ccache

WORKDIR /work

CMD ["/bin/bash"]
