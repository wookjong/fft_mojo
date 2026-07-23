#!/usr/bin/env python3
"""Inputs for a benchmark, and the answer to check it against.

    gen-input.py <bench> <cores> <dir>      writes the inputs, prints the
                                            file arguments in order
    gen-input.py --check <bench> <dir>      compares the output against the
                                            expected result

The expected result is computed here rather than in the target, so the answer
being checked does not come from the same place as the answer being produced.

The input length depends on the core count only so that the microthreads
divide evenly; the *result* must not depend on it, and that they agree across
topologies is most of what running at several is for.
"""

import struct
import sys

SEED = 20260723


def rng():
    """A generator that does not depend on the host's Python version."""
    state = SEED

    def next_u32():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        return state >> 8

    return next_u32


def write_i32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}i", *values))


def read_i32(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{len(data) // 4}i", data))


def round_up(n, m):
    return ((n + m - 1) // m) * m


# ---------------------------------------------------------------------------

def gen_vector_add(cores, d):
    """W = 8 lanes per microthread, from benchmarks/vector_add.mojo."""
    n = round_up(512, 8 * cores)
    r = rng()
    a = [(r() % 2000) - 1000 for _ in range(n)]
    b = [(r() % 2000) - 1000 for _ in range(n)]
    write_i32(f"{d}/a.bin", a)
    write_i32(f"{d}/b.bin", b)
    write_i32(f"{d}/expect.bin", [x + y for x, y in zip(a, b)])
    return ["a.bin", "b.bin", "c.bin"]


def check_vector_add(d):
    return read_i32(f"{d}/c.bin"), read_i32(f"{d}/expect.bin")


def gen_histogram(cores, d):
    """UNROLL = 16 samples per microthread; 256 bins."""
    n = round_up(1024, 16 * cores)
    r = rng()
    # Deliberately narrow, so bins collide and the atomic has to hold up.
    samples = [r() % 64 for _ in range(n)]
    write_i32(f"{d}/samples.bin", samples)

    bins = [0] * 256
    for s in samples:
        bins[s] += 1
    write_i32(f"{d}/expect.bin", bins)
    return ["samples.bin", "hist.bin"]


def check_histogram(d):
    return read_i32(f"{d}/hist.bin"), read_i32(f"{d}/expect.bin")


BENCHES = {
    "vector_add": (gen_vector_add, check_vector_add),
    "histogram": (gen_histogram, check_histogram),
}


def main(argv):
    if len(argv) >= 2 and argv[1] == "--check":
        bench, d = argv[2], argv[3]
        got, want = BENCHES[bench][1](d)
        if got == want:
            return 0
        if len(got) != len(want):
            sys.stderr.write(f"length {len(got)}, expected {len(want)}\n")
            return 1
        shown = 0
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                sys.stderr.write(f"[{i}] got {g}, expected {w}\n")
                shown += 1
                if shown == 8:
                    sys.stderr.write("...\n")
                    break
        return 1

    bench, cores, d = argv[1], int(argv[2]), argv[3]
    if bench not in BENCHES:
        sys.stderr.write(f"no inputs defined for {bench}\n")
        return 1
    files = BENCHES[bench][0](cores, d)
    # The launcher takes them in this order, after the core count and chunk.
    print(" ".join(f"{d}/{f}" for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
