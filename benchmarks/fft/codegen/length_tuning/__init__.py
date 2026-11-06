"""Per-length kernel tuning of the VkFFT-M2NDP path (adopted 2026-10-31, updated 2026-11-01).

`generate_recursive_fft_kernels(..., length_tuning=SPEC)` renders the normal
kernel and then rewrites it with the configuration measured fastest for the
plan's (N, batch) in implementations/all_lengths_20261031 (report_ko.md
rounds 1-15, report_r16_ko.md rounds 16-21, report_r22_ko.md round 22,
report_r24_ko.md / report_r25_ko.md rounds 24-25; one timing per
configuration, 28-unit / 1640 MHz simulator).

SPEC:
  'auto' (default)  the measured configuration for this (N, batch) when the
        rendered configuration is the one it was measured on (see
        `resolve_length_tuning`); otherwise the source is returned unchanged.
  'auto-ip'         as 'auto', and every tuned configuration also writes its
        result over the input buffer (IP), so that all (N, batch) pairs are in
        place, not only 16384x8. Configurations 'auto' leaves untouched stay
        untouched (out of place).
  'off'             the source unchanged, byte for byte (the previous output).
  explicit spec     a leaf spec for N <= 4096 (codegen.length_tuning.leaf, e.g.
        'D1+TR+R8x4x8+IX+PB') or 'L' / slot tokens / 'UX' for N >= 8192
        (codegen.length_tuning.slots, e.g. 'L+A1x2+F12+B2x+B4+UX'), then 'PBR',
        'AL'; either may end with 'IP'. Rejected with a reason if the
        configuration does not support it.

N <= 4096 (single leaf, task fusion off): the leaf task is replaced by 8-lane
Stockham passes (leaf.py). For a batch that was not measured the spec of the
nearest measured batch of the same N is used (O<P> scaled to ceil(batch/28)
owners; S<P> falls back to D4 when its SPAD arena would not fit).
N >= 8192 (fused grouped task): token L is the n8+f8+sp layout generalized to
far length N/64 (layout.py; with layout_rewrite='n8+f8+sp' already applied it
is skipped), then the slot tokens (slots.py), then UX (unsigned div/mod of the
non-negative index variables of the fused task), then PBR (pbr.py: per-item
base pointers for the fused task's DRAM accesses), then AL (alias.py, N=16384
grouped task only: the far leaf writes over its input buffer work1). Only the
measured (N, batch) pairs are tuned automatically.

IP (both paths, inplace.py): the result is written over the input buffer (the
host binds output to input); applied last, kernels unchanged.
"""
from __future__ import annotations

import math
import re

# Measured best per (N, batch); vs. the generator output of the same options.
#   (n, batch): (spec, best_us, base_us)
MEASURED = {
    (64, 1): ('O1+IX+PB', 7.402, 8.632),
    (64, 256): ('O10+IX+PB', 7.721, 21.844),
    (128, 1): ('D1+TR+R8x4x4+IX+PB', 7.403, 11.355),
    (256, 1): ('D1+TR+R8x4x8+IX+PB', 7.802, 11.617),
    (256, 4): ('D1+TR+R8x4x8+IX+PB', 7.970, 12.622),
    (256, 16): ('D1+TR+R8x4x8+IX+PB', 8.006, 13.549),
    (256, 64): ('S24+J+TR+R8x4x8+IX+PL+PB', 8.092, 19.276),
    (512, 1): ('D1+TR+R8x4x4x4+IX+PB', 7.987, 11.960),
    (1024, 1): ('D1+TR+R8x4x4x8+IX+PB', 8.560, 13.554),
    (1024, 4): ('D2+TR+R8x4x8x4+IX+PB', 8.586, 17.905),
    (1024, 16): ('S32+TR+R8x8x4x4+IX+PL', 8.925, 42.715),
    (1024, 64): ('S48+J+TR+R8x4x8x4+IX+W16', 11.821, 73.329),
    (2048, 1): ('D1+TR+R8x4x4x4x4+IX+PB', 8.765, 13.409),
    (4096, 1): ('D1+TR+R8x4x4x8x4+IX+W16+PB', 9.190, 16.978),
    (4096, 4): ('D4+TR+R8x4x8x4x4+IX+PL+PB', 9.635, 32.580),
    (8192, 1): ('L+A1x2+F12+B2x+B4+UX', 11.184, 21.913),
    (16384, 1): ('L+A1x2+F12+B2x+B4+UX', 13.110, 19.724),
    (16384, 8): ('L+A1x6+F12+B4+B3+UX+PBR+AL+IP', 19.130, 46.096),
}

LEAF_MAX_N = 4096
# The simulator target of every measurement (2026-10-21 refresh audit); the rewrites also assume its 28 units.
MEASURED_TARGET = dict(lmul1_float32_lanes=4, spad_capacity_bytes=122880, max_concurrent_scratchpad_bytes=65536,
                       interleave_chunk_uthreads=8, num_ndp_units=28, uthread_bytes=32, mapping_stride_bytes=256,
                       max_kernel_register=8, supports_vector_spill=False)
_SPAD_ARENA_FLOATS = 28672
# Options the leaf specs were measured with (the single-leaf benchmark path).
_LEAF_OPTIONS = dict(compute_lanes=4, narrow_middle_stages=True, spread_across_units=True, task_fusion='off')
# Options the recursive specs were measured with (the adopted fused grouped path).
_RECURSIVE_OPTIONS = dict(task_fusion='auto', task_fusion_control=False, leaf_storage='spad', boundary_layout='grouped',
                          boundary_fusion='pre', near_group=4, far_group=1, leaf_phase_policy='far12-private',
                          leaf_codegen_schedule='far0-short-live', compute_lanes=4)
_RECURSIVE_BY_BATCH = {1: dict(spad_allocation='phase-packed', far12_codelet='native4-control', batch_window='4'),
                       8: dict(spad_allocation='live-banks', far12_codelet='native4-latch-forward', batch_window='8')}


_UX = re.compile(r'(?<![*/%] )\b(gid|tile|local_tile|job) (//|%) (\d+)\b')


def _unsigned_index(source):
    """UX: unsigned div/mod of the non-negative index variables inside the fused task (no sign fix-ups)."""
    start = source.index('struct FFTFusedTask0(NDPTask):')
    end = source.index('\ndef main() raises:')
    body, count = _UX.subn(lambda m: f'Int(UInt({m[1]}) {m[2]} {m[3]})', source[start:end])
    assert count, 'no index div/mod found'
    return source[:start] + body + source[end:]


def _is_leaf(plan):
    return type(plan.root).__name__ == 'FFTLeafPlan'


def select_leaf_spec(n, batch):
    """Spec for a single-leaf (N <= 4096) plan: measured, or nearest measured batch."""
    if (n, batch) in MEASURED:
        return MEASURED[(n, batch)][0]
    measured = sorted(b for (m, b) in MEASURED if m == n)
    if not measured:
        return None
    nearest = min(measured, key=lambda b: (abs(math.log2(b) - math.log2(batch)), -b))
    spec = MEASURED[(n, nearest)][0]
    parts = spec.split('+')
    mode = re.fullmatch(r'([SOD])(\d+)', parts[0])
    nslot = -(-batch // 28)
    if mode[1] == 'O':
        parts[0] = f'O{min(64, nslot)}'
    elif mode[1] == 'S' and nslot * 4 * n > _SPAD_ARENA_FLOATS:
        parts = ['D4'] + [p for p in parts[1:] if p != 'J']
    return '+'.join(parts)


AUTO_IN_PLACE = 'auto-ip'


def resolve_length_tuning(plan, options, spec='auto'):
    """(spec or None, reason). `options` are the full render options (defaults included)."""
    if spec == AUTO_IN_PLACE:
        chosen, reason = _resolve(plan, options, 'auto')
        if chosen is None:
            return None, reason
        return (chosen if _strip_ip(chosen)[1] else chosen + '+IP'), reason + ', in place'
    return _resolve(plan, options, spec)


def _resolve(plan, options, spec):
    if spec in (None, 'off', ''):
        return None, 'length_tuning off'
    if not isinstance(spec, str):
        raise ValueError('length_tuning must be a string')
    explicit = spec != 'auto'

    def unsupported(reason):
        if explicit:
            raise ValueError(f'length_tuning {spec!r}: {reason}')
        return None, reason

    if plan.inverse:
        return unsupported('measured for forward transforms only')
    target = options.get('target')
    mismatch = {k: getattr(target, k, None) for k, v in MEASURED_TARGET.items() if getattr(target, k, None) != v}
    if mismatch:
        return unsupported(f'measured on the 28-unit target profile; this target differs in {mismatch}')
    n, batch = plan.n, plan.batch
    if _is_leaf(plan):
        if getattr(plan.root.kernel, 'kernel_name', None) != 'FFTVkfftLeaf0':
            return unsupported('leaf tuning applies to the VkFFT-M2NDP single-leaf kernel')
        if not (64 <= n <= LEAF_MAX_N and n & (n - 1) == 0):
            return unsupported(f'leaf tuning covers power-of-two N in 64..{LEAF_MAX_N}')
        # Measured on distributed execution with one stage per launch; the host of that path owns the
        # DRAM exchange buffers (buf_a/buf_b) mode D uses.
        dist = plan.root.kernel.distributed
        if dist is None or dist.fuse_stages != 1:
            return unsupported('leaf tuning was measured on distributed execution with fuse_stages=1')
        for key, value in _LEAF_OPTIONS.items():
            if options.get(key) != value:
                return unsupported(f'leaf tuning was measured with {key}={value!r} (got {options.get(key)!r})')
        if explicit:
            from . import leaf
            leaf.parse(_strip_ip(spec)[0])
            return spec, 'explicit'
        chosen = select_leaf_spec(n, batch)
        if chosen is None:
            return None, f'no measured leaf configuration for N={n}'
        return chosen, 'measured' if (n, batch) in MEASURED else 'nearest measured batch'
    for key, value in _RECURSIVE_OPTIONS.items():
        if options.get(key) != value:
            return unsupported(f'recursive tuning was measured with {key}={value!r} (got {options.get(key)!r})')
    layout = options.get('layout_rewrite', 'off')
    if layout not in ('off', 'n8+f8+sp'):
        return unsupported(f'recursive tuning starts from layout_rewrite off or n8+f8+sp (got {layout!r})')
    if explicit:
        return spec, 'explicit'
    if (n, batch) not in MEASURED:
        return None, f'no measured recursive configuration for N={n}, batch={batch}'
    required = _RECURSIVE_BY_BATCH[1 if batch == 1 else 8]
    for key, value in required.items():
        if str(options.get(key)) != value:
            return None, f'N={n} batch={batch} was measured with {key}={value!r} (got {options.get(key)!r})'
    return MEASURED[(n, batch)][0], 'measured'


def _strip_ip(spec):
    """(spec without a trailing IP, in_place)."""
    parts = spec.split('+')
    if parts[-1] == 'IP' and len(parts) > 1:
        return '+'.join(parts[:-1]), True
    return spec, False


def apply_length_tuning(source, plan, spec, options):
    """Rewrite the rendered source with SPEC; every step proves its own shape."""
    from . import inplace
    try:
        spec, in_place = _strip_ip(spec)
        if _is_leaf(plan):
            from . import leaf
            source = leaf.apply(spec, source, plan.n, plan.batch)
            return inplace.apply(source) if in_place else source
        from . import alias, layout, pbr, slots
        parts = spec.split('+')
        ux, base_pointers, aliased = 'UX' in parts, 'PBR' in parts, 'AL' in parts
        for token in ('UX', 'PBR', 'AL'):
            if token in parts:
                parts.remove(token)
        if 'L' in parts:
            parts.remove('L')
            if options.get('layout_rewrite', 'off') == 'off':
                source = layout.apply(source)
        source = slots.apply('+'.join(parts), source) if parts else source
        source = _unsigned_index(source) if ux else source
        source = pbr.apply(source) if base_pointers else source
        source = alias.apply(source) if aliased else source
        return inplace.apply(source) if in_place else source
    except (AssertionError, StopIteration, KeyError, TypeError, IndexError) as error:
        raise ValueError(f'length_tuning {spec!r} does not match the rendered kernel: {error!r}') from None
