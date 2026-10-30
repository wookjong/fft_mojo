"""Candidate A/B: near FFTs stay in SIMD lanes end to end.

Tile = 4 adjacent near FFTs r = 4g..4g+3 of batch q (g = tile % 64). The
emitted near stage0 already computes with lane = r % 4; only its stores change.

  SPAD (per tile slot, Float32):  tile_slot*512 + plane*256 + pos*4 + lane
  near output, A (work0):         q*N + g*256 + k*4 + lane
  packed twiddle (host table):    g*256 + k*4 + lane  =  W_N^(r*k)
  far input (work1, unchanged):   q*N + k*256 + r

Stage1 butterfly b (0..7, one per worker) reads positions b + 8m and writes
k = b + 8j, exactly the emitted stage1 (which handled four b in lanes).
A: MIDDLE = packed vector load, packed twiddle multiply, contiguous store of
   eight r (two packed vectors joined) into the canonical far input.
B: near stage1 multiplies by the packed twiddle and stores the far input
   directly; MIDDLE and its full-array pass are removed.
"""
from __future__ import annotations

import re

from .phases import NEAR0, NEAR1, MID, get_function, replace_function, remove_function

_STORE_RE = re.compile(r'^(\s*)FFTFusedTask0\.arena\.store\(0 \+ \(4 \* tile_slot \+ (\d)\) \* 128 \+ (0|64) \+ \((.*)\), (\w+)\[(\d)\]\)\s*$')


def _window(text):
    return 'window_base' if 'var window_base' in text else '0'


def near0_lane_major(body):
    out, seen = [], {}
    for line in body.split('\n'):
        m = _STORE_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent, slot_lane, plane, pos, name, lane = m[1], int(m[2]), int(m[3]), m[4], m[5], int(m[6])
        assert slot_lane == lane, line
        key = (name, plane, pos)
        seen.setdefault(key, set()).add(lane)
        if lane == 0:
            out.append(f'{indent}FFTFusedTask0.arena.store(tile_slot * 512 + {plane * 4} + ({pos}) * 4, {name})')
    assert all(v == {0, 1, 2, 3} for v in seen.values()) and len(seen) == 16, seen
    return '\n'.join(out)


def _near1_parts(src):
    body = get_function(src, NEAR1)
    lines = body.split('\n')
    header = []
    for line in lines:
        header.append(line)
        if line.strip() == 'var tile = gid // 8':
            break
    tiles = int(re.search(r'while tile < (\d+):', body)[1])
    # Butterfly of the work_index == 0 branch (lanes = butterflies 0..3).
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


_OUT_RE = re.compile(r'^(\s*)p\.output_(real|imag)_base\.store\(out_batch_base \+ (\d+), (\w+)\)\s*$')


def near1_lane_major(src, fused):
    header, tiles, compute = _near1_parts(src)
    window = _window('\n'.join(header))
    ind = ' ' * 12
    out = list(header)
    out += [f'        while tile < {tiles}:',
            f'{ind}var tile_slot = tile // 28',
            f'{ind}var tile_batch = tile // 64',
            f'{ind}var group = tile % 64',
            f'{ind}var b = work_slot',
            f'{ind}var sbase = tile_slot * 512 + b * 4']
    for m in range(8):
        out.append(f'{ind}var rr{m} = FFTFusedTask0.arena.load[DType.float32, 4](sbase + {32 * m})')
        out.append(f'{ind}var ii{m} = FFTFusedTask0.arena.load[DType.float32, 4](sbase + 256 + {32 * m})')
    if fused:
        out.append(f'{ind}var out_base = {window} + tile_batch * 16384 + b * 256 + group * 4')
        out.append(f'{ind}var tw_base = group * 256 + b * 4')
    else:
        out.append(f'{ind}var out_base = {window} + tile_batch * 16384 + group * 256 + b * 4')
    stored = {}
    pending = {}
    for line in compute:
        m = _OUT_RE.match(line)
        if not m:
            out.append(ind + line.strip())
            continue
        part, offset, name = m[2], int(m[3]), m[4]
        assert offset % 8 == 0 and offset < 64, line
        j = offset // 8
        stored.setdefault(j, set()).add(part)
        if not fused:
            out.append(f'{ind}p.output_{part}_base.store(out_base + {32 * j}, {name})')
            continue
        pending[(j, part)] = name
        if (j, 'real') in pending and (j, 'imag') in pending:
            xr, xi = pending.pop((j, 'real')), pending.pop((j, 'imag'))
            out += [f'{ind}var mwr{j} = p.boundary_twiddle_real_base.load[width=4](tw_base + {32 * j})',
                    f'{ind}var mwi{j} = p.boundary_twiddle_imag_base.load[width=4](tw_base + {32 * j})',
                    f'{ind}p.output_real_base.store(out_base + {2048 * j}, {xr} * mwr{j} - {xi} * mwi{j})',
                    f'{ind}p.output_imag_base.store(out_base + {2048 * j}, {xr} * mwi{j} + {xi} * mwr{j})']
    assert sorted(stored) == list(range(8)) and all(v == {'real', 'imag'} for v in stored.values()), stored
    out.append(f'{ind}tile += 28')
    return '\n'.join(out)


def middle_vector(src):
    body = get_function(src, MID)
    limit = int(re.search(r'if gid >= (\d+):', body)[1])
    window = _window(body)
    lines = ['    @staticmethod', '    def node_2_transpose():',
             '        var gid = global_uthread_id()', f'        if gid >= {limit}:', '            return',
             '        ref p = FFTFusedTask0.params[].node_1']
    if window != '0':
        lines += ['        var batch_base = Int(FFTFusedTask0.batch_base.load[DType.int32, 1](0)[0])',
                  '        var window_base = batch_base * 16384']
    ind = '        '
    lines += [f'{ind}var q = gid // 256',
              f'{ind}var t = gid % 256',
              f'{ind}var pair = t // 8',
              f'{ind}var k0 = (t % 8) * 8',
              f'{ind}var src0 = {window} + q * 16384 + pair * 512 + k0 * 4',
              f'{ind}var tw0 = pair * 512 + k0 * 4',
              f'{ind}var dst0 = {window} + q * 16384 + k0 * 256 + pair * 8']
    for kk in range(8):
        for half in (0, 1):
            off = half * 256 + kk * 4
            lines += [f'{ind}var xr{kk}_{half} = p.src_real_base.load[width=4](src0 + {off})',
                      f'{ind}var xi{kk}_{half} = p.src_imag_base.load[width=4](src0 + {off})',
                      f'{ind}var wr{kk}_{half} = p.twiddle_real_base.load[width=4](tw0 + {off})',
                      f'{ind}var wi{kk}_{half} = p.twiddle_imag_base.load[width=4](tw0 + {off})',
                      f'{ind}var yr{kk}_{half} = xr{kk}_{half} * wr{kk}_{half} - xi{kk}_{half} * wi{kk}_{half}',
                      f'{ind}var yi{kk}_{half} = xr{kk}_{half} * wi{kk}_{half} + xi{kk}_{half} * wr{kk}_{half}']
        lines += [f'{ind}p.dst_real_base.store(dst0 + {kk * 256}, yr{kk}_0.join(yr{kk}_1))',
                  f'{ind}p.dst_imag_base.store(dst0 + {kk * 256}, yi{kk}_0.join(yi{kk}_1))']
    return '\n'.join(lines)


_TW_ADDR = '            var tw_addr1 = tw_r1 * 64 + tw_c1'
_TW_PACKED = '            var tw_addr1 = (tw_r1 // 4) * 256 + tw_c1 * 4 + tw_r1 % 4'


def _pack_twiddle(src):
    assert src.count(_TW_ADDR) == 1 and 'while tw_r1 < 256:' in src and 'while tw_c1 < 64:' in src
    return src.replace(_TW_ADDR, _TW_PACKED)


def variant_A(src):
    src = replace_function(src, NEAR0, near0_lane_major(get_function(src, NEAR0)))
    src = replace_function(src, NEAR1, near1_lane_major(src, fused=False))
    src = replace_function(src, MID, middle_vector(src))
    return _pack_twiddle(src)


def variant_B(src):
    src = replace_function(src, NEAR0, near0_lane_major(get_function(src, NEAR0)))
    src = replace_function(src, NEAR1, near1_lane_major(src, fused=True))
    src = remove_function(src, MID)
    src, calls = re.subn(r'^ +launch_parallel\[FFTFusedTask0\.node_2_transpose\]\(\)\n', '', src, flags=re.M)
    assert calls == 1
    # near stage1 now writes the far input (work1) and reads the packed twiddle.
    fields = ('    var loop_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]\n')
    leaf1 = src.index('struct FFTVkfftLeaf1Params(Movable):')
    at = src.index(fields, leaf1) + len(fields)
    src = src[:at] + ('    var boundary_twiddle_real_base: UnsafePointer[Float32, MutAnyOrigin]\n'
                      '    var boundary_twiddle_imag_base: UnsafePointer[Float32, MutAnyOrigin]\n') + src[at:]
    old = 'FFTVkfftLeaf1Params(input_real, input_imag, work0_real, work0_imag, looptw0_real, looptw0_imag)'
    new = ('FFTVkfftLeaf1Params(input_real, input_imag, work1_real, work1_imag, looptw0_real, looptw0_imag, '
           'twiddle1_real, twiddle1_imag)')
    assert src.count(old) == 1
    src = src.replace(old, new)
    return _pack_twiddle(src)
