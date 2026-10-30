"""8-lane variants: one SIMD8 access fills one 32-byte memory packet.

The vector unit is VLEN=256 (8 x f32) and the build packs memory requests in
32-byte packets; the current kernels use SIMD4 (16-byte) accesses. A SIMD4
value already occupies a whole vector register, so SIMD8 keeps the register
count and doubles the work per instruction.

N8  candidate B with 8-lane tiles: tile = 8 adjacent near FFTs r = 8g..8g+7,
    lane = r % 8 for both near stages, SPAD tile_slot*1024 + plane*512 + pos*8,
    near stage1 multiplies by the 8-packed twiddle (g*512 + k*8 + lane) and
    stores 32 contiguous bytes of the far input (q*N + k*256 + 8g). MIDDLE gone.
F8  far stage0 with 8 butterflies per vector: 4 jobs per far FFT, 32-byte
    input/twiddle loads, one strided store (stride 8) per output vector into
    the unchanged far SPAD layout, so the far1+2 codelet is untouched.
"""
from __future__ import annotations

import re

from .phases import NEAR0, NEAR1, MID, FAR0, get_function, replace_function, remove_function
from . import near_lane_major as nlm


def near0_n8(body):
    tiles = int(re.search(r'while tile < (\d+):', body)[1])
    assert tiles % 64 == 0
    body = body.replace(f'while tile < {tiles}:', f'while tile < {tiles // 2}:')
    for old, new in (('var tile_batch = tile // 64', 'var tile_batch = tile // 32'),
                     ('var leaf_base = (tile % 64) * 4', 'var leaf_base = (tile % 32) * 8')):
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    loads = re.findall(r'p\.input_(?:real|imag)_base\.load\[width=4\]', body)
    assert len(loads) == 16
    body = body.replace('p.input_real_base.load[width=4]', 'p.input_real_base.load[width=8]')
    body = body.replace('p.input_imag_base.load[width=4]', 'p.input_imag_base.load[width=8]')
    out, seen = [], {}
    for line in body.split('\n'):
        m = nlm._STORE_RE.match(line)
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


def near1_n8(src):
    header, tiles, compute = nlm._near1_parts(src)
    window = nlm._window('\n'.join(header))
    ind = ' ' * 12
    out = list(header)
    out += [f'        while tile < {tiles // 2}:',
            f'{ind}var tile_slot = tile // 28',
            f'{ind}var tile_batch = tile // 32',
            f'{ind}var group = tile % 32',
            f'{ind}var b = work_slot',
            f'{ind}var sbase = tile_slot * 1024 + b * 8']
    for m in range(8):
        out.append(f'{ind}var rr{m} = FFTFusedTask0.arena.load[DType.float32, 8](sbase + {64 * m})')
        out.append(f'{ind}var ii{m} = FFTFusedTask0.arena.load[DType.float32, 8](sbase + 512 + {64 * m})')
    out.append(f'{ind}var out_base = {window} + tile_batch * 16384 + b * 256 + group * 8')
    out.append(f'{ind}var tw_base = group * 512 + b * 8')
    pending, stored = {}, set()
    for line in compute:
        m = nlm._OUT_RE.match(line)
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
                    f'{ind}p.output_real_base.store(out_base + {2048 * j}, {xr} * mwr{j} - {xi} * mwi{j})',
                    f'{ind}p.output_imag_base.store(out_base + {2048 * j}, {xr} * mwi{j} + {xi} * mwr{j})']
    assert len(stored) == 16 and not pending
    out.append(f'{ind}tile += 28')
    return '\n'.join(out)


_TW8 = '            var tw_addr1 = (tw_r1 // 8) * 512 + tw_c1 * 8 + tw_r1 % 8'


def variant_N8(src):
    original = src
    src = nlm.variant_B(src)  # MIDDLE removal, Params/host binding, packed table
    src = replace_function(src, NEAR0, near0_n8(get_function(original, NEAR0)))
    src = replace_function(src, NEAR1, near1_n8(original))
    assert src.count(nlm._TW_PACKED) == 1
    return src.replace(nlm._TW_PACKED, _TW8)


# ------------------------------------------------------------------- F8


_IN_RE = re.compile(r'load\[width=4\]\(in_batch_base \+ \(\((\d+) \+ residue_it \* 0\) \+ outer_it \* 4\)\)')
_TW_RE = re.compile(r'load\[width=4\]\(\(\((\d+) \+ residue_it \* 4\) \+ outer_it \* 4\)\)')
_ST_RE = re.compile(r'^(\s*)FFTFusedTask0\.arena\.store\(0 \+ spad_base \+ (256 \+ )?\(\((\d+) \+ residue_it \* 0\) \+ outer_it \* 32\), (\w+)\[(\d)\]\)\s*$')


def far0_f8(body):
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
    out = head + [f'        var job = work_slot',
                  f'        while job < {slots * 4}:',
                  f'            var leaf_slot = job // 4',
                  f'            var outer_it = job % 4',
                  f'            var fft_index = unit_slot + leaf_slot * 28',
                  f'            if fft_index < {ffts}:',
                  f'                var in_batch_base = {window}(fft_index * 256)',
                  f'                var spad_base = leaf_slot * 512']
    groups = {}
    count_in = count_tw = 0
    for line in lines[start:stop]:
        m = _ST_RE.match(line)
        if m:
            indent, imag, j, name, lane = m[1], bool(m[2]), int(m[3]), m[4], int(m[5])
            groups.setdefault(name, {})[lane] = j
            if lane == 0:
                plane = '256 + ' if imag else ''
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
    return replace_function(src, FAR0, far0_f8(get_function(src, FAR0)))
