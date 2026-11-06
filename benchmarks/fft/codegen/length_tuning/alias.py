"""AL: the far leaf (node 3) of the N=16384 grouped task writes over its own input buffer (round 24, 2026-11-02).

The grouped task moves the data input -> work1 (near leaf) -> work0 (far leaf) -> output (transpose):
four 1 MiB buffers at batch 8, three of them written, while the memory-side L2 holds 2.36 MB (32 x
72 KiB). From the second call on every write then evicts a dirty line of the previous call. The far
leaf reads all of work1 into SPAD in node_3_stage_0 and writes only in node_3_stage_1_2 (a later
kernel, after the launch barrier), so it may write over work1. Host binding change only.
"""

LEAF2 = 'FFTVkfftLeaf2Params(work1_real, work1_imag, work0_real, work0_imag, '
POST = 'FFTVkfftPost0Params(work0_real, work0_imag, '


def apply(src):
    assert src.count(LEAF2) == 1 and src.count(POST) == 1, 'expected one grouped host binding'
    src = src.replace(LEAF2, 'FFTVkfftLeaf2Params(work1_real, work1_imag, work1_real, work1_imag, ')
    return src.replace(POST, 'FFTVkfftPost0Params(work1_real, work1_imag, ')
