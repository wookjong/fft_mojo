"""Slot-utilization tokens for the two-level (N >= 8192) fused grouped task.

Ported from implementations/all_lengths_20261031/transforms/slots.py (the code
the 2026-10-31 slot and all-length measurements ran); only the tokens used by
the measured best configurations are kept (A3n/A3f/B5/A4 were measured slower
and dropped). They apply on top of the n8+f8+sp layout.

Tokens (combinable with '+'; applied in the fixed order of ORDER below):
  A1x2 / A1x4   G worker groups per unit (8 -> 8G uthreads) in near0, near1,
                far0 and far12: group g of a unit takes tiles/jobs k = g (mod G).
                far12 private scratch grows to 8G workers (arena enlarged).
  A1x2n/A1x4n   same, near phases only;  A1x2f/A1x4f far phases only.
  F12           far1+2 as 8-lane radix-8 + radix-4, in place in the far SPAD
                slot (no private scratch), stage twiddles as constants.
  B2            Params pointers read once into locals at phase entry.
  B3            strided SPAD stores as vsse (vp.strided.store) instead of vsoxei.
  B4            twiddles by recurrence: near1 boundary twiddle (2 loads +
                complex multiplies instead of 16 loads), far0 W256^(bj) powers.
  B4n / B4f     B4 for near1 only / far0 only.
  B2x           B2 in every phase except near0.
  A1 tokens may be combined when their phases are disjoint (A1x8n+A1x4f).
"""
from __future__ import annotations

import re

from ._codelets import _radix4, _cmul_const
from ._source import NEAR0, NEAR1, FAR0, FAR12, get_function, replace_function

ORDER = ('F12', 'B4', 'A1', 'B3', 'B2')


def _sub1(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def _arena(src):
    return int(re.search(r'comptime arena = scratchpad\[(\d+), Float32', src)[1])


def _set_arena(src, size):
    old = _arena(src)
    if size <= old:
        return src
    return re.sub(r'comptime arena = scratchpad\[\d+, Float32', f'comptime arena = scratchpad[{size}, Float32', src, count=1)


def _ensure_domain(src, uthreads):
    """MappedDomain gives one uthread per 8 pool floats; make sure gids reach `uthreads`."""
    m = re.search(r'var fusion_pool0_elems = (\d+)', src)
    need = uthreads * 8
    if int(m[1]) >= need:
        return src
    return src.replace(m[0], f'var fusion_pool0_elems = {need}', 1)


def _indent(lines, by):
    return [(' ' * by + l) if l.strip() else l for l in lines]

# ------------------------------------------------------------------ A1


def a1(src, G, which=('n', 'f')):
    lw = '(gid % 8 + 8 * (gid // 224))'
    applied = []
    if 'n' in which:
        for name in (NEAR0, NEAR1):
            try:
                body = get_function(src, name)
            except StopIteration:
                continue
            if 'var tile = gid // 8' not in body:
                continue
            body = _sub1(body, 'if gid >= 224:', f'if gid >= {224 * G}:')
            body = _sub1(body, 'var tile = gid // 8', 'var tile = (gid // 8) % 28 + 28 * (gid // 224)')
            body, n = re.subn(r'^( {12})tile \+= 28$', rf'\g<1>tile += {28 * G}', body, flags=re.M)
            assert n == 1, name
            src = replace_function(src, name, body)
            applied.append(name)
    if 'f' in which:
        for name in (FAR0, FAR12):
            body = get_function(src, name)
            if 'var unit_slot = gid // 8' not in body:
                continue
            body = _sub1(body, 'if gid >= 224:', f'if gid >= {224 * G}:')
            body = _sub1(body, 'var unit_slot = gid // 8', 'var unit_slot = (gid // 8) % 28')
            if 'var component_job = work_slot' in body:
                body = _sub1(body, 'var component_job = work_slot', 'var component_job = work_slot + 8 * (gid // 224)')
                body, n = re.subn(r'^( +)component_job \+= 8$', rf'\g<1>component_job += {8 * G}', body, flags=re.M)
                assert n == 1
                m = re.search(r'var local_base = (\d+) \+ work_slot \* (\d+)', body)
                base, per = int(m[1]), int(m[2])
                body = body.replace(m[0], f'var local_base = {base} + {lw} * {per}')
                src = _set_arena(src, base + 8 * G * per)
            else:
                body = _sub1(body, 'var job = work_slot', 'var job = work_slot + 8 * (gid // 224)')
                body, n = re.subn(r'^( +)job \+= 8$', rf'\g<1>job += {8 * G}', body, flags=re.M)
                assert n == 1, name
            src = replace_function(src, name, body)
            applied.append(name)
    assert applied, 'A1 found no phase to widen'
    return _ensure_domain(src, 224 * G)

# ------------------------------------------------------------------ F12


def _dft8(x, p):
    """Forward DFT8 on complex names x[0..7]: radix-4 over m2 for m1 in {0,1}, W8 twiddles, radix-2."""
    lines, a = [], {}
    for m1 in (0, 1):
        l, ys = _radix4([x[m1], x[m1 + 2], x[m1 + 4], x[m1 + 6]], f'{p}h{m1}')
        lines += l
        for k2 in range(4):
            a[m1, k2] = ys[k2]
    outs = [None] * 8
    for k2 in range(4):
        lines += _cmul_const(a[1, k2], k2, 8, f'{p}w{k2}')
        br, bi = f'{p}w{k2}r', f'{p}w{k2}i'
        ar, ai = a[0, k2]
        lines += [f'var {p}o{k2}r = {ar} + {br}', f'var {p}o{k2}i = {ai} + {bi}',
                  f'var {p}o{k2 + 4}r = {ar} - {br}', f'var {p}o{k2 + 4}i = {ai} - {bi}']
        outs[k2] = (f'{p}o{k2}r', f'{p}o{k2}i')
        outs[k2 + 4] = (f'{p}o{k2 + 4}r', f'{p}o{k2 + 4}i')
    return lines, outs


def f12(src):
    """far12 of an a-point far FFT (a = N/64, M = a/8 points per far12 FFT, O = M/8 column groups):
    DFT8 over m (stride M) per column group o, twiddle W_M^(o*k1), radix-O over o, in place in SPAD."""
    n = int(re.search(r'^comptime N = (\d+)', src, re.M)[1])
    a = n // 64
    M, O = a // 8, a // 64
    assert O in (2, 4), ('F12 supports far lengths 128 and 256', a)
    old = get_function(src, FAR12)
    ffts = int(re.search(r'if fft_index < (\d+):', old)[1])
    slots = -(-ffts // 28)
    out_expr = re.search(r'var out_batch_base = (.*)$', old, re.M)[1]
    head = []
    for line in old.split('\n'):
        if 'var local_base' in line or 'var component_job' in line:
            continue
        head.append(line)
        if line.strip() == 'var unit_slot = gid // 8':
            break
    ind = ' ' * 16
    out = head + ['        var job = work_slot', f'        while job < {slots}:',
                  '            var fft_index = unit_slot + job * 28', f'            if fft_index < {ffts}:',
                  f'{ind}# F12 fft body begin',
                  f'{ind}var spad_base = job * {2 * a}', f'{ind}var out_batch_base = {out_expr}']
    body = []
    for o in range(O):
        x = []
        for m in range(8):
            body += [f'var s{o}x{m}r = FFTFusedTask0.arena.load[DType.float32, 8](spad_base + {o * 8 + M * m})',
                     f'var s{o}x{m}i = FFTFusedTask0.arena.load[DType.float32, 8](spad_base + {a} + {o * 8 + M * m})']
            x.append((f's{o}x{m}r', f's{o}x{m}i'))
        l, ys = _dft8(x, f's{o}')
        body += l
        for k1 in range(8):
            body += _cmul_const(ys[k1], o * k1, M, f's{o}t{k1}')
            body += [f'FFTFusedTask0.arena.store(spad_base + {o * 8 + M * k1}, s{o}t{k1}r)',
                     f'FFTFusedTask0.arena.store(spad_base + {a} + {o * 8 + M * k1}, s{o}t{k1}i)']
    out += [ind + l for l in body]
    out += [f'{ind}var k1 = 0', f'{ind}while k1 < 8:']
    inner, z = [], []
    for o in range(O):
        inner += [f'var z{o}r = FFTFusedTask0.arena.load[DType.float32, 8](spad_base + {o * 8} + {M} * k1)',
                  f'var z{o}i = FFTFusedTask0.arena.load[DType.float32, 8](spad_base + {a} + {o * 8} + {M} * k1)']
        z.append((f'z{o}r', f'z{o}i'))
    if O == 4:
        l, ys = _radix4(z, 'q')
    else:
        (ar, ai), (br, bi) = z
        l = [f'var qy0r = {ar} + {br}', f'var qy0i = {ai} + {bi}', f'var qy1r = {ar} - {br}', f'var qy1i = {ai} - {bi}']
        ys = [('qy0r', 'qy0i'), ('qy1r', 'qy1i')]
    inner += l
    for k2 in range(O):
        inner += [f'p.output_real_base.store(out_batch_base + 8 * k1 + {64 * k2}, {ys[k2][0]})',
                  f'p.output_imag_base.store(out_batch_base + 8 * k1 + {64 * k2}, {ys[k2][1]})']
    inner.append('k1 += 1')
    out += [ind + '    ' + l for l in inner]
    out += [f'{ind}# F12 fft body end', '            job += 8']
    return replace_function(src, FAR12, '\n'.join(out))

# ------------------------------------------------------------------ B2


def b2(src, skip=()):
    names = re.findall(r'^    def (\w+)\((?:gid: Int)?\):', src, re.M)
    for name in names:
        if name in skip:
            continue
        body = get_function(src, name)
        m = re.search(r'^(\s*)ref p = FFTFusedTask0\.params\[\]\.(\w+)$', body, re.M)
        if m is None:
            continue
        fields = sorted(set(re.findall(r'\bp\.(\w+)', body)))
        lines = body.split('\n')
        out = []
        for line in lines:
            if line == m[0]:
                out.append(line)
                out += [f'{m[1]}var p_{f} = p.{f}' for f in fields]
                continue
            out.append(re.sub(r'\bp\.(\w+)', r'p_\1', line))
        src = replace_function(src, name, '\n'.join(out))
    return src

# ------------------------------------------------------------------ B3


_STRIDED = re.compile(r'^(\s*)\(FFTFusedTask0\.arena\.ptr\(\) \+ \((.*)\)\)\.strided_store\[width=(\d+)\]\((\w+), (\d+)\)\s*$')


def b3(src):
    out, n = [], 0
    for line in src.split('\n'):
        m = _STRIDED.match(line)
        if m:
            ind, addr, w, val, stride = m[1], m[2], int(m[3]), m[4], int(m[5])
            out.append(f'{ind}llvm_intrinsic["llvm.experimental.vp.strided.store", NoneType, has_side_effect=True]'
                       f'({val}, FFTFusedTask0.arena.ptr() + ({addr}), Int64({4 * stride}), SIMD[DType.bool, {w}](fill=True), Int32({w}))')
            n += 1
        else:
            out.append(line)
    assert n, 'no strided stores to convert'
    src = '\n'.join(out)
    return 'from std.sys.intrinsics import llvm_intrinsic\n' + src

# ------------------------------------------------------------------ B4


def _cmul(a, b, out):
    return [f'var {out}r = {a}r * {b}r - {a}i * {b}i', f'var {out}i = {a}r * {b}i + {a}i * {b}r']


def _function_with(src, needle):
    names = [n for n in re.findall(r'^    def (\w+)\((?:gid: Int)?\):', src, re.M) if needle in get_function(src, n)]
    assert len(names) == 1, (needle, names)
    return names[0]


def b4(src, which=('n', 'f')):
    if 'n' in which:
        src = _b4_near(src)
    if 'f' in which:
        src = _b4_far(src)
    return src


def _b4_near(src):
    # near1: boundary twiddle W^(r(b+8j)) = W^(rb) * (W^(8r))^j
    near_name = _function_with(src, 'p.boundary_twiddle_real_base.load')
    body = get_function(src, near_name)
    rx = re.compile(r'^(\s*)var mw([ri])(\d) = p\.boundary_twiddle_(real|imag)_base\.load\[width=(\d+)\]\(tw_base \+ (\d+)\)$')
    lines, out, done = body.split('\n'), [], set()
    width = None
    plan = {0: None, 4: ('0', 's4'), 1: ('0', 's1'), 5: ('4', 's1'), 2: ('1', 's1'), 6: ('5', 's1'), 3: ('2', 's1'), 7: ('6', 's1')}
    for line in lines:
        m = rx.match(line)
        if not m:
            out.append(line)
            continue
        ind, part, j, width = m[1], m[2], int(m[3]), int(m[5])
        if part == 'i':
            continue
        if j == 0:
            lanes = width
            out += [f'{ind}var mw0r = p.boundary_twiddle_real_base.load[width={lanes}](tw_base)',
                    f'{ind}var mw0i = p.boundary_twiddle_imag_base.load[width={lanes}](tw_base)',
                    f'{ind}var s1r = p.boundary_twiddle_real_base.load[width={lanes}](tw_base - b * {lanes} + {8 * lanes})',
                    f'{ind}var s1i = p.boundary_twiddle_imag_base.load[width={lanes}](tw_base - b * {lanes} + {8 * lanes})']
            out += [ind + l for l in _cmul('s1', 's1', 's2') + _cmul('s2', 's2', 's4')]
        else:
            src_j, step = plan[j]
            assert int(src_j) in done, ('recurrence order', j)
            out += [ind + l for l in _cmul(f'mw{src_j}', step, f'mw{j}')]
        done.add(j)
        out.append(None)
    out = [l for l in out if l is not None]
    assert done == set(range(8)), done
    text = '\n'.join(out)
    text = re.sub(r'\bmwr(\d)\b', r'mw\1r', text)
    text = re.sub(r'\bmwi(\d)\b', r'mw\1i', text)
    return replace_function(src, near_name, text)


def _b4_far(src):
    # far0: W256^(b j) = (W256^b)^j with lanes b
    far_name = _function_with(src, 'var twr1 = p.loop_twiddle_real_base.load[width=8]')
    body = get_function(src, far_name)
    rx = re.compile(r'^(\s*)var tw([ri])(\d) = p\.loop_twiddle_(real|imag)_base\.load\[width=(\d+)\]\(\((\d+) \+ (\w+) \* (\d+)\)\)$')
    bases = {}
    for line in body.split('\n'):
        m = rx.match(line)
        if m:
            bases[int(m[3])] = (int(m[6]), int(m[5]), int(m[8]), m[7])
    assert sorted(bases) == list(range(1, 8)), bases
    b1, lanes, step, loopvar = bases[1]
    out, have = [], set()
    formula = {2: ('w1', 'w1'), 3: ('w2', 'w1'), 4: ('w2', 'w2'), 5: ('w4', 'w1'), 6: ('w4', 'w2'), 7: ('w4', 'w3')}
    need = {2: {1}, 3: {1, 2}, 4: {2}, 5: {1, 4}, 6: {2, 4}, 7: {3, 4}}

    def make(j, ind):
        lines = []
        if j in have:
            return lines
        if j == 1:
            lines += [f'{ind}var w1r = p.loop_twiddle_real_base.load[width={lanes}](({b1} + {loopvar} * {step}))',
                      f'{ind}var w1i = p.loop_twiddle_imag_base.load[width={lanes}](({b1} + {loopvar} * {step}))']
        else:
            for d in sorted(need[j]):
                lines += make(d, ind)
            a, c = formula[j]
            lines += [ind + l for l in _cmul(a, c, f'w{j}')]
        have.add(j)
        return lines
    for line in body.split('\n'):
        m = rx.match(line)
        if not m:
            out.append(line)
            continue
        ind, part, j = m[1], m[2], int(m[3])
        if part == 'r':
            out += make(j, ind)
            out.append(f'{ind}var twr{j} = w{j}r')
        else:
            out.append(f'{ind}var twi{j} = w{j}i')
    return replace_function(src, far_name, '\n'.join(out))

# ------------------------------------------------------------------ apply


def tokens(spec):
    parts = spec.split('+')
    a1s = [p for p in parts if p.startswith('A1')]
    phases = [c for p in a1s for c in (p[4:] or 'nf')]
    assert len(phases) == len(set(phases)), 'A1 tokens must cover disjoint phases'
    known = set(ORDER) | {'A1', 'B2x', 'B4n', 'B4f'}
    for p in parts:
        assert (p.startswith('A1') and re.fullmatch(r"A1x[2-8][nf]?", p)) or p in known, p
    return parts


def apply(spec, src):
    parts = tokens(spec)
    for key in ORDER:
        if key == 'A1':
            for p in parts:
                if p.startswith('A1'):
                    m = re.fullmatch(r'A1x(\d)([nf]?)', p)
                    src = a1(src, int(m[1]), (m[2],) if m[2] else ('n', 'f'))
        elif key == 'B4' and ('B4n' in parts or 'B4f' in parts):
            src = b4(src, tuple(w for w in 'nf' if f'B4{w}' in parts or 'B4' in parts))
        elif key == 'B2' and 'B2x' in parts:
            src = b2(src, skip=(NEAR0,))
        elif key in parts:
            src = dict(F12=f12, B2=b2, B3=b3, B4=b4)[key](src)
    return src
