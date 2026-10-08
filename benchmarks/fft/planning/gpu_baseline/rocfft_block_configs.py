"""Generic rocFFT SBCC/SBRC launch configurations at the pinned commit.

Source: rocm-libraries bee97df517907c771de17189cb867d3c401285ae,
projects/rocfft/library/src/device/kernels/configs/config_{sbcc,sbrc}.py.
Launch sizing follows device/kernel-generator.py and generator/stockham_gen.cpp,
stockham_gen_base.h: generic 64 KiB LDS, occupancy two, and 16-byte elements
(the same generated kernel key supports double precision).
"""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class BlockConfig:
    workgroup_size: int
    threads_per_transform: int
    factors: tuple[int, ...]
    scheme: str
    half_lds: bool = False
    direct_to_from_reg: bool = True
    use_3steps_large_twd: bool = False  # single-precision setting


def _config(length, factors, scheme, *, workgroup_size=None,
            threads_per_transform=None, flavour="uwide", half_lds=False,
            direct_to_from_reg=True, use_3steps_large_twd=False):
    factors = tuple(factors)
    assert prod(factors) == length
    if workgroup_size is None:
        workgroup_size = 16 * length // min(factors)
    if half_lds:
        workgroup_size = min(1024, workgroup_size * 2)
    if threads_per_transform is None:
        divisor = max(factors) if flavour == "wide" else min(factors)
        threads_per_transform = length // divisor
    bytes_per_batch = length * 16 // (2 if half_lds else 1)
    transforms_per_block = min(32768 // bytes_per_batch,
                               workgroup_size // threads_per_transform)
    assert transforms_per_block > 0
    return BlockConfig(threads_per_transform * transforms_per_block,
                       threads_per_transform, factors, scheme, half_lds,
                       direct_to_from_reg, use_3steps_large_twd)


def _cc(length, factors, **kwargs):
    return _config(length, factors, "CS_KERNEL_STOCKHAM_BLOCK_CC", **kwargs)


def _rc(length, factors, **kwargs):
    return _config(length, factors, "CS_KERNEL_STOCKHAM_BLOCK_RC", **kwargs)


SBCC_TABLE = {
    50: _cc(50, (10, 5), workgroup_size=256, use_3steps_large_twd=True),
    52: _cc(52, (13, 4), use_3steps_large_twd=True),
    60: _cc(60, (6, 10)),
    64: _cc(64, (8, 8), workgroup_size=256, use_3steps_large_twd=True),
    72: _cc(72, (8, 3, 3), use_3steps_large_twd=True),
    80: _cc(80, (10, 8)),
    81: _cc(81, (3, 3, 3, 3), use_3steps_large_twd=True),
    84: _cc(84, (7, 2, 6), threads_per_transform=14, use_3steps_large_twd=True),
    96: _cc(96, (8, 3, 4), workgroup_size=256),
    100: _cc(100, (5, 5, 4), workgroup_size=100, half_lds=True, use_3steps_large_twd=True),
    104: _cc(104, (13, 8), use_3steps_large_twd=True),
    108: _cc(108, (6, 6, 3), use_3steps_large_twd=True),
    112: _cc(112, (4, 7, 4)),
    121: _cc(121, (11, 11), workgroup_size=128, use_3steps_large_twd=True),
    125: _cc(125, (5, 5, 5), use_3steps_large_twd=True),
    128: _cc(128, (16, 8), workgroup_size=256, threads_per_transform=16, use_3steps_large_twd=True),
    160: _cc(160, (4, 10, 4), flavour="wide"),
    168: _cc(168, (7, 6, 4), workgroup_size=128, half_lds=True, use_3steps_large_twd=True),
    169: _cc(169, (13, 13), workgroup_size=256, use_3steps_large_twd=True),
    192: _cc(192, (8, 6, 4), use_3steps_large_twd=True),
    200: _cc(200, (5, 8, 5)),
    208: _cc(208, (13, 16)),
    216: _cc(216, (6, 6, 6), threads_per_transform=36),
    224: _cc(224, (8, 7, 4), use_3steps_large_twd=True),
    240: _cc(240, (8, 5, 6)),
    243: _cc(243, (3, 3, 3, 3, 3), workgroup_size=243, use_3steps_large_twd=True),
    256: _cc(256, (8, 4, 8), flavour="wide", use_3steps_large_twd=True),
    280: _cc(280, (8, 5, 7)),
    289: _cc(289, (17, 17), use_3steps_large_twd=True),
    336: _cc(336, (6, 7, 8)),
    343: _cc(343, (7, 7, 7), use_3steps_large_twd=True),
    512: _cc(512, (8, 8, 8), use_3steps_large_twd=True),
}

SBRC_TABLE = {
    17: _rc(17, (17,), workgroup_size=256, threads_per_transform=1),
    49: _rc(49, (7, 7), workgroup_size=196, threads_per_transform=7),
    50: _rc(50, (10, 5), workgroup_size=50, threads_per_transform=5, direct_to_from_reg=False),
    64: _rc(64, (4, 4, 4), workgroup_size=128, threads_per_transform=16),
    81: _rc(81, (3, 3, 3, 3), workgroup_size=243, threads_per_transform=27),
    100: _rc(100, (5, 5, 4), workgroup_size=100, threads_per_transform=25),
    112: _rc(112, (4, 7, 4), workgroup_size=448, threads_per_transform=28),
    121: _rc(121, (11, 11), workgroup_size=128, threads_per_transform=11),
    125: _rc(125, (5, 5, 5), workgroup_size=250, threads_per_transform=25),
    128: _rc(128, (8, 4, 4), workgroup_size=128, threads_per_transform=16),
    169: _rc(169, (13, 13), workgroup_size=256, threads_per_transform=13),
    192: _rc(192, (6, 4, 4, 2), workgroup_size=256, threads_per_transform=32),
    200: _rc(200, (8, 5, 5), workgroup_size=400, threads_per_transform=40),
    243: _rc(243, (3, 3, 3, 3, 3), workgroup_size=256, threads_per_transform=27),
    256: _rc(256, (4, 4, 4, 4), workgroup_size=256, threads_per_transform=32),
    289: _rc(289, (17, 17), workgroup_size=128, threads_per_transform=17),
    343: _rc(343, (7, 7, 7), workgroup_size=256, threads_per_transform=49),
    512: _rc(512, (8, 8, 8), workgroup_size=512, threads_per_transform=128),
    625: _rc(625, (5, 5, 5, 5), workgroup_size=128, threads_per_transform=125),
    1331: _rc(1331, (11, 11, 11), workgroup_size=256, threads_per_transform=121),
}
