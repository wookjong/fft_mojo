#!/usr/bin/env python3
"""Generate a launcher and a check for a compiled M²NDP kernel.

Up to now the simulator has run hand-written assembly. This runs what the
compiler actually produces, which means implementing the launcher half of the
contract in docs/INTERFACE.md:

  - the scratchpad region is the launcher's to provide. `__m2ndp_spad_size`
    comes from the link script and says how much of it the globals want, so
    the base pointer goes at `region + __m2ndp_spad_size` and the arguments
    are written from there upwards.
  - the identity values arrive in registers, one per value, listed below.
  - microthreads are run one at a time. The contract has no barrier and
    combining happens through atomics, so a sequential schedule is a legal
    one -- see docs/SIMULATION.md for what that means it cannot catch.

The kernel preserves nothing, so the loop keeps its counter in memory rather
than in a register.

Expected results are computed here, in Python, so the answer being checked
does not come from the same place as the answer being produced.

    ./sim/gen-kernel-test.py vector_add > test.s
"""

import struct
import sys

# docs/INTERFACE.md, "How the IDs arrive". The assignment is provisional and
# lives in RISCVM2ndpArgInfo.h; this is the only other place that knows it.
ARG_REGS = {
    "scratchpad_base": "a0",
    "offset": "a1",
    "addr": "a2",
    "ndp_id": "a3",
    "local_uthread_id": "a4",
    "global_uthread_id": "a5",
    "group_size": "a6",
    "group_id": "a7",
}

out = []
def emit(s=""):
    out.append(s)


def prologue():
    emit("        .text")
    emit("        .globl _start")
    emit("_start:")
    emit("        # Bare metal starts with both units off.")
    emit("        li      t0, (1 << 13) | (1 << 9)   # FS, VS = Initial")
    emit("        csrs    mstatus, t0")
    emit()


def scratchpad_base(reg):
    """base = region + __m2ndp_spad_size.

    __m2ndp_spad_size is an absolute symbol, so %hi/%lo of it yield the value
    rather than an address.
    """
    emit(f"        la      {reg}, spad_region")
    emit(f"        lui     t0, %hi(__m2ndp_spad_size)")
    emit(f"        addi    t0, t0, %lo(__m2ndp_spad_size)")
    emit(f"        add     {reg}, {reg}, t0")


def launch(kernel, args, nthreads, group_size):
    """Run `kernel` once per microthread, arguments already in place."""
    tag = kernel
    emit(f"        # ---- {kernel}: {nthreads} microthreads")
    scratchpad_base("t1")
    for i, sym in enumerate(args):
        emit(f"        la      t2, {sym}")
        emit(f"        sd      t2, {i * 8}(t1)")
    emit(f"        la      t0, uthread_counter")
    emit(f"        sd      zero, 0(t0)")
    emit(f"{tag}_loop:")
    scratchpad_base(ARG_REGS['scratchpad_base'])
    emit(f"        la      t0, uthread_counter")
    emit(f"        ld      t1, 0(t0)")
    emit(f"        li      {ARG_REGS['offset']}, 0")
    emit(f"        li      {ARG_REGS['addr']}, 0")
    emit(f"        li      {ARG_REGS['ndp_id']}, 0")
    emit(f"        li      t2, {group_size}")
    emit(f"        remu    {ARG_REGS['local_uthread_id']}, t1, t2")
    emit(f"        mv      {ARG_REGS['global_uthread_id']}, t1")
    emit(f"        li      {ARG_REGS['group_size']}, {group_size}")
    emit(f"        li      {ARG_REGS['group_id']}, 0")
    emit(f"        call    {kernel}")
    emit(f"        # The kernel preserves nothing, so the counter lives in memory.")
    emit(f"        la      t0, uthread_counter")
    emit(f"        ld      t1, 0(t0)")
    emit(f"        addi    t1, t1, 1")
    emit(f"        sd      t1, 0(t0)")
    emit(f"        li      t2, {nthreads}")
    emit(f"        blt     t1, t2, {tag}_loop")
    emit()


def check_words(sym, expected, width=4):
    """Compare `len(expected)` words at `sym`, accumulating into t6."""
    load = {4: "lw", 8: "ld"}[width]
    for i, want in enumerate(expected):
        emit(f"        la      t0, {sym}")
        emit(f"        {load}      t1, {i * width}(t0)")
        emit(f"        li      t2, {want}")
        emit(f"        xor     t1, t1, t2")
        emit(f"        or      t6, t6, t1")


def epilogue():
    emit("        # Exit 0 only if every check matched.")
    emit("        snez    a0, t6")
    emit("        slli    a0, a0, 1")
    emit("        ori     a0, a0, 1")
    emit("        la      t0, tohost")
    emit("        sd      a0, 0(t0)")
    emit("1:      j       1b")
    emit()


def common_data(spad_bytes=4096):
    emit("        .bss")
    emit("        .align 6")
    emit("uthread_counter: .dword 0")
    emit("        .align 6")
    emit("        # The launcher owns this: .spad only reserves a size.")
    emit(f"spad_region: .zero {spad_bytes}")
    emit()
    emit('        .section .tohost,"aw",@progbits')
    emit("        .align 6")
    emit("        .globl tohost")
    emit("tohost:   .dword 0")
    emit("        .align 6")
    emit("        .globl fromhost")
    emit("fromhost: .dword 0")


# ---------------------------------------------------------------------------

def gen_vector_add():
    W = 8              # int32 lanes per microthread, from the kernel
    THREADS = 8
    N = W * THREADS

    a = [i for i in range(N)]
    b = [3 * i + 1 for i in range(N)]
    c = [(x + y) & 0xffffffff for x, y in zip(a, b)]

    prologue()
    emit("        li      t6, 0                      # difference accumulator")
    emit()
    launch("vector_add", ["buf_a", "buf_b", "buf_c"], THREADS, group_size=4)
    check_words("buf_c", c)
    epilogue()

    emit("        .data")
    emit("        .align 6")
    emit("buf_a:  .word " + ", ".join(str(v) for v in a))
    emit("        .align 6")
    emit("buf_b:  .word " + ", ".join(str(v) for v in b))
    emit("        .align 6")
    emit("buf_c:  .word " + ", ".join("0" for _ in c))
    emit()
    common_data()


def gen_histogram():
    BINS = 256
    UNROLL = 16        # samples per microthread, from the kernel
    THREADS = 8
    GROUP = 8
    N = UNROLL * THREADS

    # Samples chosen to collide: several microthreads hit the same bins.
    samples = [(i * 7) % 12 for i in range(N)]

    bins = [0] * BINS
    for s in samples:
        bins[s] += 1

    prologue()
    emit("        li      t6, 0                      # difference accumulator")
    emit()
    # Three phases, in order. The scratchpad survives across them, which is
    # the property being exercised as much as the atomics are.
    launch("histogram_init", [], GROUP, group_size=GROUP)
    launch("histogram_body", ["buf_samples"], THREADS, group_size=GROUP)
    launch("histogram_final", ["buf_out"], GROUP, group_size=GROUP)
    # Only the bins that were touched, plus a couple that were not.
    check_words("buf_out", bins[:16])
    epilogue()

    emit("        .data")
    emit("        .align 6")
    emit("buf_samples: .word " + ", ".join(str(v) for v in samples))
    emit("        .align 6")
    emit("buf_out: .word " + ", ".join("0" for _ in range(BINS)))
    emit()
    common_data()


GENERATORS = {
    "vector_add": gen_vector_add,
    "histogram": gen_histogram,
}

if len(sys.argv) != 2 or sys.argv[1] not in GENERATORS:
    sys.stderr.write(f"usage: {sys.argv[0]} [{'|'.join(GENERATORS)}]\n")
    sys.exit(2)

emit(f"# GENERATED by sim/gen-kernel-test.py {sys.argv[1]} -- do not edit.")
emit("#")
emit("# Launches the compiled kernel the way the contract says a task is")
emit("# launched, then checks the result against values computed in Python.")
emit()
GENERATORS[sys.argv[1]]()
sys.stdout.write("\n".join(out) + "\n")
