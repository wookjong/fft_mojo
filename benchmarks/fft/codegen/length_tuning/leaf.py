"""Vectorized single-leaf FFT task for N <= 4096 (replaces the generated leaf task).

Ported from implementations/all_lengths_20261031/transforms/leafgen.py (the
code the 2026-10-31 / 2026-11-01 all-length measurements ran). Only the tokens
used by the measured best configurations are kept; H, K, U, E, M, radix 16/32
(report_ko.md) and U2, TRH, AOS, SP, four-step (report_r16_ko.md) were measured
slower, without effect or not executable on the simulator, and dropped.

The single-leaf VkFFT-M2NDP kernel uses 2-4 lane vectors, per-lane scalar
stores and a DRAM round trip between stages, with few busy uthreads. This
renders the same transform with the techniques adopted for N=16384:
  * 8-lane Stockham passes (radix 8, last pass radix 2/4 when log2 N % 3 != 0):
    every load is a contiguous 8-float vector; stores are contiguous except
    pass 0 (one strided store per output vector, vsse via vp.strided.store);
  * exchange between passes in the unit's SPAD (mode S/O) or DRAM (mode D);
  * configurable uthreads per FFT (A1 analogue) or one owner uthread per FFT
    running all passes in a single launch (mode O);
  * pass twiddles from a table built for this layout (pass 1 has Ns=8: one
    hoisted vector per r), optionally by recurrence w_r = w_1^r (B4 analogue).

Spec tokens ('+'-joined):
  S<P>  SPAD exchange, one launch per pass; FFT f lives on unit f % 28, P
        uthreads of that unit share each pass.
  O<P>  SPAD, single launch; each uthread owns whole FFTs (P owners per unit).
  D<G>  DRAM exchange (Params buf_a/buf_b), one launch per pass, 224*G workers
        over all (FFT, butterfly) items.
  TR    twiddle recurrence (load w_1 only).
  R<a>x<b>[x...]  radix plan (each 2, 4 or 8, first 8 so every later pass has
        Ns % 8 == 0). A radix-4 pass costs about half a radix-8 pass, so plans
        such as 8x4x4x8 beat the default 8..8 + 2/4.
  J     mode S: distribute (FFT slot, vector block) items jointly over the P
        uthreads instead of looping FFT slots inside every uthread.
  IX    index math by shift/mask (all strides are powers of two) and unsigned
        unit/slot decomposition, instead of signed // and % with sign fix-ups.
  W16   radix <= 4 passes whose lanes stay inside one twiddle block run on
        16-lane vectors (LMUL 2): half the iterations.
  PL    pass 0 stores contiguously (element i at (i % 8) * L0 + i // 8) and
        pass 1 reads with strided loads, instead of strided pass-0 stores.
  PB    per-item base pointers: ptr = buffer + (base + item part) once per
        butterfly, every contiguous access is ptr.load/store(constant offset)
        (effective in modes D/O; LLVM folds it back in mode S).
The host main keeps its buffers and Params; only the twiddle table, the
launch domain and the task struct are replaced.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ._codelets import _radix4
from .slots import _dft8

UNITS, V = 28, 8
EXTRA = ('IX', 'W16', 'PL', 'PB')
TASK = 'FFTVkfftLeaf0'


def extras(spec):
    toks = spec.split('+')
    return {t: t in toks for t in EXTRA}


def radices(n, plan=None):
    m = int(math.log2(n))
    assert 1 << m == n and 64 <= n <= 4096, n
    if plan is None:
        return [8] * (m // 3) + ({0: [], 1: [2], 2: [4]}[m % 3])
    assert math.prod(plan) == n and plan[0] >= 8, (n, plan)
    assert all(r in (2, 4, 8) and (n // r) % V == 0 for r in plan), (n, plan)
    return list(plan)


def passes(n, plan=None):
    out, ns = [], 1
    for r in radices(n, plan):
        out.append(dict(R=r, Ns=ns, L=n // r, VB=n // r // V))
        ns *= r
    return out


def twiddle_table(n, plan=None):
    """Per pass p >= 1: block[(r-1)*Ns + k] = W_{Ns*R}^{r*k}. Returns (offsets, complex128 array)."""
    offsets, parts, pos = [], [], 0
    for ps in passes(n, plan):
        offsets.append(pos)
        if ps['Ns'] == 1:
            continue
        r = np.arange(1, ps['R'])[:, None]
        k = np.arange(ps['Ns'])[None, :]
        block = np.exp(-2j * np.pi * r * k / (ps['Ns'] * ps['R'])).reshape(-1)
        parts.append(block)
        pos += block.size
    table = np.concatenate(parts) if parts else np.ones(1, dtype=complex)
    return offsets, table


def _hex(table):
    re32 = table.real.astype(np.float32).view(np.uint32)
    im32 = table.imag.astype(np.float32).view(np.uint32)
    return ''.join(f'{a:08x}{b:08x}' for a, b in zip(re32, im32))


def host_twiddles(n, plan=None):
    _, table = twiddle_table(n, plan)
    size = table.size
    return [f'    var looptw0_real = cxl_alloc[Float32]({size})',
            f'    var looptw0_imag = cxl_alloc[Float32]({size})',
            f'    var looptw0_real_hex = String("{_hex(table)}")',
            f'    for tw_idx in range({size}):',
            '        if True:',
            '            var tw_bits = UInt32(0)',
            '            for tw_nibble in range(8):',
            '                var tw_code = Int(looptw0_real_hex.unsafe_ptr()[tw_idx * 16 + 0 + tw_nibble])',
            '                var tw_digit = tw_code & 15',
            '                if tw_code >= 97:',
            '                    tw_digit += 9',
            '                tw_bits = (tw_bits << 4) | UInt32(tw_digit)',
            '            looptw0_real.bitcast[UInt32]()[tw_idx] = tw_bits',
            '        if True:',
            '            var tw_bits = UInt32(0)',
            '            for tw_nibble in range(8):',
            '                var tw_code = Int(looptw0_real_hex.unsafe_ptr()[tw_idx * 16 + 8 + tw_nibble])',
            '                var tw_digit = tw_code & 15',
            '                if tw_code >= 97:',
            '                    tw_digit += 9',
            '                tw_bits = (tw_bits << 4) | UInt32(tw_digit)',
            '            looptw0_imag.bitcast[UInt32]()[tw_idx] = tw_bits']


def _dft(x, p):
    if len(x) == 8:
        return _dft8(x, p)
    if len(x) == 4:
        return _radix4(x, p)
    (ar, ai), (br, bi) = x
    return ([f'var {p}y0r = {ar} + {br}', f'var {p}y0i = {ai} + {bi}',
             f'var {p}y1r = {ar} - {br}', f'var {p}y1i = {ai} - {bi}'], [(f'{p}y0r', f'{p}y0i'), (f'{p}y1r', f'{p}y1i')])


def _cmul(a, b, out):
    return [f'var {out}r = {a}r * {b}r - {a}i * {b}i', f'var {out}i = {a}r * {b}i + {a}i * {b}r']


class Space:
    """Where a pass reads or writes: DRAM input/output, a SPAD buffer, or a DRAM work buffer."""

    def __init__(self, kind, n, base=None, buf=None):
        self.kind, self.n, self.base, self.buf = kind, n, base, buf

    def load(self, part, idx, lanes=8):
        if self.kind == 'in':
            return f'p.input_{part}_base.load[width={lanes}]({self.base} + {idx})'
        off = '' if part == 'real' else f' + {self.n}'
        if self.kind == 'spad':
            return f'{TASK}.arena.load[DType.float32, {lanes}]({self.base}{off} + {idx})'
        return f'p.{self.buf}.load[width={lanes}]({self.base}{off} + {idx})'

    def _ptr(self):
        return f'{TASK}.arena.ptr()' if self.kind == 'spad' else f'p.{self.buf}'

    def pointer_lines(self, name, var):
        """PB: (lines, refs) with refs[part] = (pointer name, constant element offset of the part)."""
        if self.kind in ('in', 'out'):
            io = 'input' if self.kind == 'in' else 'output'
            lines = [f'var {name}{c} = p.{io}_{part}_base + ({self.base} + {var})'
                     for part, c in (('real', 'r'), ('imag', 'i'))]
            return lines, {'real': (f'{name}r', 0), 'imag': (f'{name}i', 0)}
        return [f'var {name} = {self._ptr()} + ({self.base} + {var})'], {'real': (name, 0), 'imag': (name, self.n)}

    @staticmethod
    def pload(refs, name, const, lanes):
        return [f'var {name}{c} = {refs[part][0]}.load[width={lanes}]({const + refs[part][1]})'
                for part, c in (('real', 'r'), ('imag', 'i'))]

    @staticmethod
    def pstore(refs, yr, yi, const):
        return [f'{refs[part][0]}.store({const + refs[part][1]}, {y})' for part, y in (('real', yr), ('imag', yi))]

    def load_lines(self, name, idx, lanes=8, stride=None):
        """Lines defining {name}r / {name}i."""
        if stride is not None:
            assert self.kind in ('spad', 'dram')
            out = []
            for part, c in (('real', 'r'), ('imag', 'i')):
                off = '' if part == 'real' else f' + {self.n}'
                out.append(f'var {name}{c} = llvm_intrinsic["llvm.experimental.vp.strided.load", '
                           f'SIMD[DType.float32, {lanes}], has_side_effect=False]({self._ptr()} + ({self.base}{off} + {idx}), '
                           f'Int64({4 * stride}), SIMD[DType.bool, {lanes}](fill=True), Int32({lanes}))')
            return out
        return [f'var {name}r = {self.load("real", idx, lanes)}', f'var {name}i = {self.load("imag", idx, lanes)}']

    def store(self, part, idx, value, stride=None, lanes=8):
        if self.kind == 'out':
            assert stride is None
            return f'p.output_{part}_base.store({self.base} + {idx}, {value})'
        off = '' if part == 'real' else f' + {self.n}'
        addr = f'{self.base}{off} + {idx}'
        if self.kind == 'spad':
            if stride is None:
                return f'{TASK}.arena.store({addr}, {value})'
            ptr = f'{TASK}.arena.ptr()'
        else:
            if stride is None:
                return f'p.{self.buf}.store({addr}, {value})'
            ptr = f'p.{self.buf}'
        return (f'llvm_intrinsic["llvm.experimental.vp.strided.store", NoneType, has_side_effect=True]'
                f'({value}, {ptr} + ({addr}), Int64({4 * stride}), SIMD[DType.bool, {lanes}](fill=True), Int32({lanes}))')

    def store_lines(self, yr, yi, idx, stride=None, lanes=8):
        return [self.store('real', idx, yr, stride, lanes), self.store('imag', idx, yi, stride, lanes)]


def _twiddle_lines(ps, offset, recurrence, k0, w='w', lanes=8):
    """Lines defining {w}{r}r/{w}{r}i (r = 1..R-1) for lanes k0..k0+lanes-1 of this pass."""
    R, Ns = ps['R'], ps['Ns']
    lines = []

    def load(r):
        idx = f'{offset + (r - 1) * Ns} + {k0}'
        return [f'var {w}{r}r = p.loop_twiddle_real_base.load[width={lanes}]({idx})',
                f'var {w}{r}i = p.loop_twiddle_imag_base.load[width={lanes}]({idx})']
    if not recurrence:
        for r in range(1, R):
            lines += load(r)
        return lines
    lines += load(1)
    have = {1}
    for r in range(2, R):
        a, b = (r // 2, r - r // 2)
        assert a in have and b in have
        lines += _cmul(f'{w}{a}', f'{w}{b}', f'{w}{r}')
        have.add(r)
    return lines


def load_inputs(ps, src, j0='j0'):
    """(lines, names) loading the R input vectors of one butterfly."""
    R, L = ps['R'], ps['L']
    lanes = ps.get('lanes', V)
    lines, xs = [], []
    if ps.get('pb') and not ps.get('pl_load'):
        lines, refs = src.pointer_lines('pin', j0)
        for r in range(R):
            lines += Space.pload(refs, f'x{r}', r * L, lanes)
            xs.append((f'x{r}r', f'x{r}i'))
        return lines, xs
    for r in range(R):
        if ps.get('pl_load'):  # previous pass stored element i at (i % 8) * L0 + i // 8
            lines += src.load_lines(f'x{r}', f'({j0} >> 3) + {r * L // 8}', lanes, stride=ps['pl_load'])
        else:
            lines += src.load_lines(f'x{r}', f'{j0} + {r * L}', lanes)
        xs.append((f'x{r}r', f'x{r}i'))
    return lines, xs


def butterfly(ps, src, dst, *, twiddles_inline, offset, recurrence, j0='j0', w='w', ix=False):
    """Body for one vector butterfly (lanes j0..j0+lanes-1) of pass ps."""
    R, Ns, L = ps['R'], ps['Ns'], ps['L']
    lanes = ps.get('lanes', V)
    lines = []
    if Ns >= V:
        if ix:
            lines += [f'var blk = {j0} >> {Ns.bit_length() - 1}', f'var k0 = {j0} & {Ns - 1}']
        else:
            lines += [f'var blk = {j0} // {Ns}', f'var k0 = {j0} % {Ns}']
    loads, xs = load_inputs(ps, src, j0)
    lines += loads
    if Ns > 1:
        if twiddles_inline:
            lines += _twiddle_lines(ps, offset, recurrence, 'k0', lanes=lanes)
        for r in range(1, R):
            lines += _cmul(f'x{r}', f'{w}{r}', f't{r}')
            xs[r] = (f't{r}r', f't{r}i')
    body, ys = _dft(xs, 'd')
    lines += body
    if ps.get('pb') and (Ns > 1 or ps.get('pl_store')):
        var = j0 if Ns == 1 else f'blk * {Ns * R} + k0'
        plines, refs = dst.pointer_lines('pout', var)
        lines += plines
        for q, (yr, yi) in enumerate(ys):
            lines += Space.pstore(refs, yr, yi, q * L if Ns == 1 else q * Ns)
        return lines
    for q, (yr, yi) in enumerate(ys):
        if Ns == 1 and ps.get('pl_store'):
            idx, stride = f'{q * L} + {j0}', None
        elif Ns == 1:
            idx, stride = f'{j0} * {R} + {q}', R
        else:
            idx, stride = f'blk * {Ns * R} + {q * Ns} + k0', None
        lines += dst.store_lines(yr, yi, idx, stride, lanes)
    return lines


def _indent(lines, k):
    return [' ' * k + l for l in lines]


def _sh(value):
    assert value & (value - 1) == 0, value
    return value.bit_length() - 1


def parse(spec):
    """Returns (mode, count, recurrence, plan, joint)."""
    mode, count, rec, plan, joint = None, None, False, None, False
    for tok in spec.split('+'):
        m = re.fullmatch(r'([SOD])(\d+)', tok)
        if m:
            assert mode is None, spec
            mode, count = m[1], int(m[2])
        elif tok == 'TR':
            rec = True
        elif tok == 'J':
            joint = True
        elif tok in EXTRA:
            pass
        elif re.fullmatch(r'R\d+(x\d+)+', tok):
            assert plan is None, spec
            plan = [int(r) for r in tok[1:].split('x')]
        else:
            raise ValueError(f'unknown leaf token {tok!r} in {spec!r}')
    if mode is None:
        raise ValueError(f'leaf spec {spec!r} needs one of S<P>, O<P>, D<G>')
    if joint and mode != 'S':
        raise ValueError('J applies to mode S')
    return mode, count, rec, plan, joint


def render(n, batch, spec):
    """Returns (struct_text, arena_floats, domain_uthreads)."""
    mode, count, rec, plan, joint = parse(spec)
    ex = extras(spec)
    ix = ex['IX']
    unit_line = 'var unit = Int(UInt(gid >> 3) % 28)' if ix else 'var unit = (gid // 8) % 28'
    t_line = 'var t = (gid & 7) + 8 * Int(UInt(gid) // 224)' if ix else 'var t = gid % 8 + 8 * (gid // 224)'
    ps_all = passes(n, plan)
    if ex['W16']:  # radix <= 4 passes whose lanes stay inside one twiddle block
        ps_all = [dict(p, lanes=16, VB=p['L'] // 16) if p['R'] <= 4 and (p['Ns'] == 1 or p['Ns'] >= 16)
                  and p['L'] % 16 == 0 else p for p in ps_all]
        assert any(p.get('lanes') == 16 for p in ps_all), 'W16: no eligible pass'
    if ex['PB']:
        ps_all = [dict(p, pb=True) for p in ps_all]
    if ex['PL']:
        assert ps_all[0]['Ns'] == 1 and ps_all[0]['R'] == 8 and ps_all[1]['Ns'] == 8
        ps_all[0] = dict(ps_all[0], pl_store=True)
        ps_all[1] = dict(ps_all[1], pl_load=ps_all[0]['L'])
    offsets, _ = twiddle_table(n, plan)
    last = len(ps_all) - 1
    methods, launches = [], []
    head = [f'ref p = {TASK}.params[]', 'var gid = global_uthread_id()']
    if mode in 'SO':
        nslot = -(-batch // UNITS)
        if mode == 'S':
            assert count <= 64
            arena = nslot * 4 * n
            domain = 224 * -(-count // 8)
        else:
            owners = count
            assert owners <= 64
            arena = owners * 4 * n
            domain = 224 * -(-owners // 8)
        assert arena <= 28672, ('SPAD arena too large', arena)
    else:
        arena = 0
        domain = 224 * count

    def spaces(pi, fbase):
        src = Space('in', n, 'in_base') if pi == 0 else (
            Space('spad', n, f'{fbase} + {2 * n * ((pi - 1) % 2)}') if mode in 'SO' else
            Space('dram', n, 'fb', 'buf_a' if (pi - 1) % 2 == 0 else 'buf_b'))
        dst = Space('out', n, 'out_base') if pi == last else (
            Space('spad', n, f'{fbase} + {2 * n * (pi % 2)}') if mode in 'SO' else
            Space('dram', n, 'fb', 'buf_a' if pi % 2 == 0 else 'buf_b'))
        return src, dst

    def hoisted(ps, pi):
        # Ns == 8: every butterfly uses lanes k = 0..7 -> one twiddle vector per r for the whole pass.
        return ps['Ns'] == V

    if mode == 'S':
        for pi, ps in enumerate(ps_all):
            src, dst = spaces(pi, 'sbase')
            hoist = hoisted(ps, pi)
            VB, lanes = ps['VB'], ps.get('lanes', V)
            body = butterfly(ps, src, dst, twiddles_inline=ps['Ns'] > 1 and not hoist, offset=offsets[pi],
                             recurrence=rec, ix=ix)
            fn = head + [unit_line, t_line, f'if t >= {count}:', '    return']
            if hoist:
                fn += _twiddle_lines(ps, offsets[pi], rec, '0')
            if joint:
                fs = f'item >> {_sh(VB)}' if ix else f'item // {VB}'
                vb = f'item & {VB - 1}' if ix else f'item % {VB}'
                fn += ['var item = t', f'while item < {nslot * VB}:', f'    var fslot = {fs}',
                       f'    var vb = {vb}', '    var f = unit + 28 * fslot',
                       f'    if f >= {batch}:', '        break',
                       f'    var in_base = f * {n}', f'    var out_base = f * {n}', f'    var sbase = fslot * {4 * n}',
                       f'    var j0 = vb * {lanes}']
                fn += _indent(body, 4) + [f'    item += {count}']
            else:
                fn += ['var fslot = 0', f'while fslot < {nslot}:', '    var f = unit + 28 * fslot',
                       f'    if f >= {batch}:', '        break',
                       f'    var in_base = f * {n}', f'    var out_base = f * {n}', f'    var sbase = fslot * {4 * n}',
                       '    var vb = t', f'    while vb < {VB}:', f'        var j0 = vb * {lanes}']
                fn += _indent(body, 8) + [f'        vb += {count}', '    fslot += 1']
            methods.append((f'pass_{pi}', fn))
            launches.append(f'pass_{pi}')
    elif mode == 'O':
        fn = head + ['var unit = (gid // 8) % 28', 'var t = gid % 8 + 8 * (gid // 224)',
                     f'if t >= {count}:', '    return', f'var sbase = t * {4 * n}']
        fn += ['var s = 0', 'while True:', f'    var f = unit + 28 * (t + {count} * s)',
               f'    if f >= {batch}:', '        break',
               f'    var in_base = f * {n}', f'    var out_base = f * {n}']
        for pi, ps in enumerate(ps_all):
            src, dst = spaces(pi, 'sbase')
            body = butterfly(ps, src, dst, twiddles_inline=ps['Ns'] > 1, offset=offsets[pi], recurrence=rec)
            fn += [f'    var vb{pi} = 0', f'    while vb{pi} < {ps["VB"]}:', f'        var j0 = vb{pi} * 8']
            fn += _indent(body, 8) + [f'        vb{pi} += 1']
        fn += ['    s += 1']
        methods.append(('run_all', fn))
        launches.append('run_all')
    else:
        workers = 224 * count
        for pi, ps in enumerate(ps_all):
            src, dst = spaces(pi, None)
            hoist = hoisted(ps, pi)
            VB, lanes = ps['VB'], ps.get('lanes', V)
            body = butterfly(ps, src, dst, twiddles_inline=ps['Ns'] > 1 and not hoist, offset=offsets[pi],
                             recurrence=rec, ix=ix)
            fn = head[:]
            if hoist:
                fn += _twiddle_lines(ps, offsets[pi], rec, '0')
            fe = f'item >> {_sh(VB)}' if ix else f'item // {VB}'
            ve = f'item & {VB - 1}' if ix else f'item % {VB}'
            fn += ['var item = gid', f'while item < {batch * VB}:', f'    var f = {fe}',
                   f'    var vb = {ve}', f'    var in_base = f * {n}', f'    var out_base = f * {n}',
                   f'    var fb = f * {2 * n}', f'    var j0 = vb * {lanes}']
            fn += _indent(body, 4) + [f'    item += {workers}']
            methods.append((f'pass_{pi}', fn))
            launches.append(f'pass_{pi}')
    out = [f'struct {TASK}(NDPTask):', f'    comptime Params = {TASK}Params']
    if arena:
        out.append(f'    comptime arena = scratchpad[{arena}, Float32, name="{TASK.lower()}_arena"]()')
    out.append('')
    for name, fn in methods:
        out += ['    @staticmethod', f'    def {name}():'] + _indent(fn, 8) + ['']
    out += ['    @staticmethod', '    def device_main():'] + [f'        launch_parallel[{TASK}.{l}]()' for l in launches]
    return '\n'.join(out) + '\n', arena, domain


def apply(spec, src, n, batch):
    """Replace the generated leaf task and the host twiddle table / launch domain."""
    struct_text, arena, domain = render(n, batch, spec)
    lines = src.split('\n')
    start = lines.index(f'struct {TASK}(NDPTask):')
    host = next(i for i in range(start, len(lines)) if lines[i].startswith('# HOST MAIN'))
    end = host - 1
    assert lines[end].startswith('# ====='), lines[end]
    lines[start:end] = struct_text.split('\n')
    text = '\n'.join(lines)
    # twiddle table block: from the looptw0_real allocation up to the pool setup
    tw = re.search(r'^    var looptw0_real = cxl_alloc\[Float32\]\(\d+\)\n(?:.*\n)*?(?=    var pool0_elems = )', text, re.M)
    assert tw, 'twiddle block not found'
    text = text[:tw.start()] + '\n'.join(host_twiddles(n, parse(spec)[3])) + '\n' + text[tw.end():]
    old = int(re.search(r'^    var pool0_elems = (\d+)$', text, re.M)[1])
    text = re.sub(r'^    var pool0_elems = \d+$', f'    var pool0_elems = {domain * 8}', text, count=1, flags=re.M)
    assert text.count(f'PooledRange.over(pool0, {old})') == 1
    text = text.replace(f'PooledRange.over(pool0, {old})', f'PooledRange.over(pool0, {domain * 8})')
    if 'llvm_intrinsic[' in text:
        text = 'from std.sys.intrinsics import llvm_intrinsic\n' + text
    return text
