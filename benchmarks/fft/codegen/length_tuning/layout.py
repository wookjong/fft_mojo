"""n8+f8+sp layout rewrite generalized to any far length a = N/64 (token L).

Ported from implementations/all_lengths_20261031/transforms/layout_general.py
(the code the 2026-10-31 N=8192 measurements ran). Derived from
codegen/layout_rewrite (wide.py, near_lane_major.py), which is verified only
for N=16384 (a=256). Every constant that scales with a
or N is computed from the source; the shape assertions are kept. For N=16384
apply() reproduces the adopted generator output byte for byte.
"""
from __future__ import annotations

import re

from ._source import NEAR0, NEAR1, MID, FAR0, POST, get_function, replace_function, remove_function, strided_phase

_STORE_RE = re.compile(r'^(\s*)FFTFusedTask0\.arena\.store\(0 \+ \(4 \* tile_slot \+ (\d)\) \* 128 \+ (0|64) \+ \((.*)\), (\w+)\[(\d)\]\)\s*$')
_OUT_RE = re.compile(r'^(\s*)p\.output_(real|imag)_base\.store\(out_batch_base \+ (\d+), (\w+)\)\s*$')
_TW_ADDR = '            var tw_addr1 = tw_r1 * 64 + tw_c1'
_TW8 = '            var tw_addr1 = (tw_r1 // 8) * 512 + tw_c1 * 8 + tw_r1 % 8'


def sizes(src):
    n = int(re.search(r'^comptime N = (\d+)', src, re.M)[1])
    assert n % 64 == 0 and n // 64 >= 64
    return n, n // 64


def _window(text):
    return 'window_base' if 'var window_base' in text else '0'


def _near1_parts(src):
    body = get_function(src, NEAR1)
    lines = body.split('\n')
    header = []
    for line in lines:
        header.append(line)
        if line.strip() == 'var tile = gid // 8':
            break
    tiles = int(re.search(r'while tile < (\d+):', body)[1])
    start = next(i for i, l in enumerate(lines) if l.strip() == 'if work_index == 0:')
    stop = next(i for i, l in enumerate(lines) if l.strip() == 'if work_index == 1:')
    branch = lines[start + 1:stop]
    loads = [l for l in branch if 'FFTFusedTask0.arena.load' in l]
    assert len(loads) == 16
    for m_, l in enumerate(loads):
        pos = 8 * (m_ // 2)
        plane = '0 + spad_base + 64 + ' if m_ % 2 else '0 + spad_base + '
        assert f'({plane}{pos})' in l, l
    first = next(i for i, l in enumerate(branch) if '# fixed radix-8 butterfly' in l)
    compute = [l for l in branch[first:] if l.strip()]
    return header, tiles, compute


def near0_n8(body, n, a):
    tiles = int(re.search(r'while tile < (\d+):', body)[1])
    assert tiles % (a // 4) == 0 and tiles % 2 == 0
    body = body.replace(f'while tile < {tiles}:', f'while tile < {tiles // 2}:')
    for old, new in ((f'var tile_batch = tile // {a // 4}', f'var tile_batch = tile // {a // 8}'),
                     (f'var leaf_base = (tile % {a // 4}) * 4', f'var leaf_base = (tile % {a // 8}) * 8')):
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    loads = re.findall(r'p\.input_(?:real|imag)_base\.load\[width=4\]', body)
    assert len(loads) == 16
    body = body.replace('p.input_real_base.load[width=4]', 'p.input_real_base.load[width=8]')
    body = body.replace('p.input_imag_base.load[width=4]', 'p.input_imag_base.load[width=8]')
    out, seen = [], {}
    for line in body.split('\n'):
        m = _STORE_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent, slot_lane, plane, pos, name, lane = m[1], int(m[2]), int(m[3]), m[4], m[5], int(m[6])
        assert slot_lane == lane
        seen.setdefault((name, plane, pos), set()).add(lane)
        if lane == 0:
            out.append(f'{indent}FFTFusedTask0.arena.store(tile_slot * 1024 + {plane * 8} + ({pos}) * 8, {name})')
    assert all(v == {0, 1, 2, 3} for v in seen.values()) and len(seen) == 16
    return '\n'.join(out)


def near1_n8(src, n, a):
    header, tiles, compute = _near1_parts(src)
    window = _window('\n'.join(header))
    ind = ' ' * 12
    out = list(header)
    out += [f'        while tile < {tiles // 2}:',
            f'{ind}var tile_slot = tile // 28',
            f'{ind}var tile_batch = tile // {a // 8}',
            f'{ind}var group = tile % {a // 8}',
            f'{ind}var b = work_slot',
            f'{ind}var sbase = tile_slot * 1024 + b * 8']
    for m in range(8):
        out.append(f'{ind}var rr{m} = FFTFusedTask0.arena.load[DType.float32, 8](sbase + {64 * m})')
        out.append(f'{ind}var ii{m} = FFTFusedTask0.arena.load[DType.float32, 8](sbase + 512 + {64 * m})')
    out.append(f'{ind}var out_base = {window} + tile_batch * {n} + b * {a} + group * 8')
    out.append(f'{ind}var tw_base = group * 512 + b * 8')
    pending, stored = {}, set()
    for line in compute:
        m = _OUT_RE.match(line)
        if not m:
            out.append(ind + line.strip())
            continue
        part, offset, name = m[2], int(m[3]), m[4]
        j = offset // 8
        pending[(j, part)] = name
        stored.add((j, part))
        if (j, 'real') in pending and (j, 'imag') in pending:
            xr, xi = pending.pop((j, 'real')), pending.pop((j, 'imag'))
            out += [f'{ind}var mwr{j} = p.boundary_twiddle_real_base.load[width=8](tw_base + {64 * j})',
                    f'{ind}var mwi{j} = p.boundary_twiddle_imag_base.load[width=8](tw_base + {64 * j})',
                    f'{ind}p.output_real_base.store(out_base + {8 * a * j}, {xr} * mwr{j} - {xi} * mwi{j})',
                    f'{ind}p.output_imag_base.store(out_base + {8 * a * j}, {xr} * mwi{j} + {xi} * mwr{j})']
    assert len(stored) == 16 and not pending
    out.append(f'{ind}tile += 28')
    return '\n'.join(out)


def _remove_middle(src, a):
    src = remove_function(src, MID)
    src, calls = re.subn(r'^ +launch_parallel\[FFTFusedTask0\.node_2_transpose\]\(\)\n', '', src, flags=re.M)
    assert calls == 1
    fields = '    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]\n'
    leaf1 = src.index('struct FFTVkfftLeaf1Params(Movable):')
    at = src.index(fields, leaf1) + len(fields)
    src = src[:at] + ('    var boundary_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]\n'
                      '    var boundary_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]\n') + src[at:]
    old = 'FFTVkfftLeaf1Params(input_real, input_imag, work0_real, work0_imag, looptw0_real, looptw0_imag)'
    new = ('FFTVkfftLeaf1Params(input_real, input_imag, work1_real, work1_imag, looptw0_real, looptw0_imag, '
           'twiddle1_real, twiddle1_imag)')
    assert src.count(old) == 1
    src = src.replace(old, new)
    assert src.count(_TW_ADDR) == 1 and f'while tw_r1 < {a}:' in src and 'while tw_c1 < 64:' in src
    return src.replace(_TW_ADDR, _TW8)


def variant_N8(src):
    n, a = sizes(src)
    original = src
    # Same order as the adopted path: near0 lane-major stores, then the n8 rewrite of both near phases.
    src = replace_function(src, NEAR0, near0_n8(get_function(original, NEAR0), n, a))
    src = replace_function(src, NEAR1, near1_n8(original, n, a))
    return _remove_middle(src, a)


_IN_RE = re.compile(r'load\[width=4\]\(in_batch_base \+ \(\((\d+) \+ residue_it \* 0\) \+ outer_it \* 4\)\)')
_TW_RE = re.compile(r'load\[width=4\]\(\(\((\d+) \+ residue_it \* 4\) \+ outer_it \* 4\)\)')


def far0_f8(body, n, a):
    st_re = re.compile(r'^(\s*)FFTFusedTask0\.arena\.store\(0 \+ spad_base \+ (' + str(a)
                       + r' \+ )?\(\((\d+) \+ residue_it \* 0\) \+ outer_it \* 32\), (\w+)\[(\d)\]\)\s*$')
    jobs = a // 64
    lines = body.split('\n')
    ffts = int(re.search(r'while fft_index < (\d+):', body)[1])
    slots = -(-ffts // 28)
    head = []
    for line in lines:
        if line.strip() == 'var fft_index = unit_slot':
            break
        head.append(line)
    start = next(i for i, l in enumerate(lines) if l.strip() == 'var residue_it = stage_job % 1') + 1
    stop = next(i for i, l in enumerate(lines) if l.strip() == 'stage_job += 8')
    window = 'window_base + ' if 'var window_base' in body else ''
    out = head + ['        var job = work_slot',
                  f'        while job < {slots * jobs}:',
                  f'            var leaf_slot = job // {jobs}',
                  f'            var outer_it = job % {jobs}',
                  '            var fft_index = unit_slot + leaf_slot * 28',
                  f'            if fft_index < {ffts}:',
                  f'                var in_batch_base = {window}(fft_index * {a})',
                  f'                var spad_base = leaf_slot * {2 * a}']
    groups = {}
    count_in = count_tw = 0
    for line in lines[start:stop]:
        m = st_re.match(line)
        if m:
            indent, imag, j, name, lane = m[1], bool(m[2]), int(m[3]), m[4], int(m[5])
            groups.setdefault(name, {})[lane] = j
            if lane == 0:
                plane = f'{a} + ' if imag else ''
                out.append(f'{indent}(FFTFusedTask0.arena.ptr() + (spad_base + {plane}({j} + outer_it * 64))).strided_store[width=8]({name}, 8)')
            continue
        if line.strip().startswith('#'):
            continue
        line, n1 = _IN_RE.subn(r'load[width=8](in_batch_base + (\1 + outer_it * 8))', line)
        line, n2 = _TW_RE.subn(r'load[width=8]((\1 + outer_it * 8))', line)
        count_in += n1
        count_tw += n2
        assert 'residue_it' not in line and 'width=4' not in line, line
        out.append(line)
    assert count_in == 16 and count_tw == 14, (count_in, count_tw)
    for name, lanes in groups.items():
        j0 = lanes[0]
        assert lanes == {l: j0 + 8 * l for l in range(4)}, (name, lanes)
    assert len(groups) == 16
    out.append('            job += 8')
    return '\n'.join(out)


def variant_F8(src):
    n, a = sizes(src)
    return replace_function(src, FAR0, far0_f8(get_function(src, FAR0), n, a))


def apply(src):
    """n8 + f8 + sp, in the order the measured N=16384 sources were produced."""
    src = variant_N8(src)
    src = variant_F8(src)
    return strided_phase(src, POST)
