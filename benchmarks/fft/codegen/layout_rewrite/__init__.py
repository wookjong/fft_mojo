"""Opt-in data-layout rewrites of the fused grouped VkFFT task (2026-10-31).

`generate_recursive_fft_kernels(..., layout_rewrite=SPEC)` renders the normal
kernel and then rewrites named phases of that exact source. SPEC is 'off'
(default, output unchanged) or '+'-joined tokens:

  n8   near FFTs in 8 SIMD lanes end to end (tile = 8 adjacent near FFTs);
       near stage1 multiplies by an 8-packed boundary twiddle and stores the
       far input directly (32-byte stores), so the MIDDLE phase is removed.
  a    near FFTs in 4 SIMD lanes end to end; MIDDLE becomes a vector twiddle
       pass over the lane-packed near output (no SPAD transpose).
  f8   far stage0 with 8 butterflies per vector, one strided SPAD store per
       output vector; the far SPAD layout (and far1+2 codelet) is unchanged.
  sn/sm/sf/sp  replace per-lane scalar SPAD stores with one strided store per
       vector in near stage0 / MIDDLE / far stage0 / POST. s = all four.

Selected path (measured 2026-10-30, implementations/layout_candidates_20261030):
  'n8+f8+sp'  B1: 25.536 -> 19.724 us;  B8 with batch_window=8: 82.234 -> 46.096 us.

Rewrites are applied in a fixed order (near, f8, strided) independent of the
token order, which is the order the measured sources were produced in.
"""
from __future__ import annotations

from .phases import NEAR0, MID, FAR0, POST, strided_phase

TOKENS = ('n8', 'a', 'f8', 's', 'sn', 'sm', 'sf', 'sp')
SELECTED = 'n8+f8+sp'


def normalize_layout_rewrite(spec):
    """Canonical token tuple for SPEC; () for 'off'. Raises ValueError."""
    if spec in (None, 'off', ''):
        return ()
    if not isinstance(spec, str):
        raise ValueError('layout_rewrite must be a string')
    tokens = [t.strip().lower() for t in spec.split('+')]
    if any(t not in TOKENS for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError(f'layout_rewrite tokens must be distinct members of {TOKENS}, got {spec!r}')
    parts = set(tokens)
    if 's' in parts:
        if parts & {'sn', 'sm', 'sf', 'sp'}:
            raise ValueError("'s' already contains sn/sm/sf/sp")
        parts = (parts - {'s'}) | {'sn', 'sm', 'sf', 'sp'}
    if {'n8', 'a'} <= parts:
        raise ValueError('n8 and a are alternative near layouts')
    if parts & {'n8', 'a'} and parts & {'sn', 'sm'}:
        raise ValueError('n8/a already store near stage0 (and MIDDLE) as vectors; drop sn/sm')
    if {'f8', 'sf'} <= parts:
        raise ValueError('f8 already uses strided far stage0 stores; drop sf')
    order = ('n8', 'a', 'f8', 'sn', 'sm', 'sf', 'sp')
    return tuple(t for t in order if t in parts)


def check_layout_rewrite_support(plan, options):
    """Reject configurations the rewrites were not built and verified for."""
    required = dict(task_fusion='auto', leaf_storage='spad', boundary_layout='grouped', boundary_fusion='pre',
                    near_group=4, far_group=1, leaf_phase_policy='far12-private',
                    leaf_codegen_schedule='far0-short-live', compute_lanes=4, task_fusion_control=False)
    for key, value in required.items():
        if options.get(key) != value:
            raise ValueError(f'layout_rewrite requires {key}={value!r} (got {options.get(key)!r})')
    if options.get('spad_allocation') not in ('live-banks', 'phase-packed'):
        raise ValueError('layout_rewrite requires spad_allocation=live-banks or phase-packed')
    if plan.n != 16384 or plan.inverse:
        raise ValueError('layout_rewrite is verified for the forward N=16384 grouped VkFFT task only')
    root = plan.root
    if getattr(root, 'a', None) != 256 or getattr(root, 'b', None) != 64:
        raise ValueError('layout_rewrite expects the 64 x 256 near/far decomposition')


def apply_layout_rewrite(source, spec):
    """Rewrite the rendered fused-task source; every step proves its own shape."""
    from . import near_lane_major, wide
    steps = dict(n8=wide.variant_N8, a=near_lane_major.variant_A, f8=wide.variant_F8,
                 sn=lambda s: strided_phase(s, NEAR0), sm=lambda s: strided_phase(s, MID),
                 sf=lambda s: strided_phase(s, FAR0), sp=lambda s: strided_phase(s, POST))
    for token in normalize_layout_rewrite(spec):
        try:
            source = steps[token](source)
        except (AssertionError, StopIteration, KeyError, TypeError) as error:
            raise ValueError(f'layout_rewrite token {token!r} does not match the rendered kernel: {error!r}') from None
    return source
