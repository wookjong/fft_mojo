"""Kernel-source structure helpers for length tuning (phase lookup, address
algebra, strided SPAD stores).

Ported unchanged from implementations/all_lengths_20261031/transforms/variants.py,
the code the 2026-10-31 all-length measurements ran (its function lookup also
accepts `@always_inline` roles and `def name(gid: Int)` as emitted by the slot
transforms). Only the parts the adopted tokens use are kept.
"""
from __future__ import annotations

import random
import re

# ---------------------------------------------------------------- structure

_DEF_RE = re.compile(r'^    def (\w+)\((?:gid: Int)?\):\s*$')


def function_span(src_lines, name):
    """Line span [start, end) of `    @staticmethod / def name():` incl. decorator."""
    start = next(i for i, l in enumerate(src_lines) if _DEF_RE.match(l) and _DEF_RE.match(l)[1] == name)
    while src_lines[start - 1].startswith('    @'):
        start -= 1
    assert src_lines[start].strip() == '@staticmethod'
    start += 1
    end = start + 1
    while end < len(src_lines):
        line = src_lines[end]
        if line.startswith('    @') or (line and not line.startswith(' ')):
            break
        end += 1
    # Leave trailing comment/blank lines (FFT_EXECUTION records) outside.
    while src_lines[end - 1].strip() == '' or src_lines[end - 1].startswith('#'):
        end -= 1
    return start - 1, end


def get_function(src, name):
    lines = src.split('\n')
    a, b = function_span(lines, name)
    return '\n'.join(lines[a:b])


def replace_function(src, name, new_text):
    lines = src.split('\n')
    a, b = function_span(lines, name)
    return '\n'.join(lines[:a] + new_text.rstrip('\n').split('\n') + lines[b:])


def remove_function(src, name):
    lines = src.split('\n')
    a, b = function_span(lines, name)
    return '\n'.join(lines[:a] + lines[b:])

# ---------------------------------------------------------- address algebra


_IDENT_RE = re.compile(r'[A-Za-z_]\w*')


def evaluate(expr, env):
    names = set(_IDENT_RE.findall(expr))
    scope = {n: env.setdefault(n, random.Random(hash(n) & 0xffff).randint(3, 997) * 7 + 1) for n in names}
    return eval(expr, {'__builtins__': {}}, scope)


def environments(count=4, seed=11):
    rng = random.Random(seed)
    return [_Env(rng.randint(0, 1 << 30)) for _ in range(count)]


class _Env(dict):
    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = self.rng.randint(1, 4000)
        return self[key]


_SCALAR_STORE_RE = re.compile(r'^(\s*)FFTFusedTask0\.arena\.store\((.*), (\w+)\[(\d+)\]\)\s*$')


def _runs(lines):
    """Maximal runs of consecutive per-lane scalar arena stores."""
    i = 0
    while i < len(lines):
        if _SCALAR_STORE_RE.match(lines[i]):
            j = i
            while j < len(lines) and _SCALAR_STORE_RE.match(lines[j]):
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def _groups(lines, a, b):
    groups = {}
    for k in range(a, b):
        m = _SCALAR_STORE_RE.match(lines[k])
        indent, addr, name, lane = m[1], m[2], m[3], int(m[4])
        g = groups.setdefault(name, dict(indent=indent, lanes={}, first=k))
        assert lane not in g['lanes'], lines[k]
        g['lanes'][lane] = addr
    return groups


def _stride(lanes, envs):
    width = len(lanes)
    assert sorted(lanes) == list(range(width)), lanes
    strides = set()
    for env in envs:
        values = [evaluate(lanes[l], env) for l in range(width)]
        strides |= {values[l + 1] - values[l] for l in range(width - 1)}
    assert len(strides) == 1, ('non-uniform lane stride', lanes)
    return strides.pop()


def strided_rewrite(body):
    """Replace every complete per-lane scalar store group by one strided store."""
    lines = body.split('\n')
    envs = environments()
    out, cursor, count = [], 0, 0
    for a, b in list(_runs(lines)):
        out.extend(lines[cursor:a])
        groups = _groups(lines, a, b)
        for name, g in sorted(groups.items(), key=lambda kv: kv[1]['first']):
            stride = _stride(g['lanes'], envs)
            width = len(g['lanes'])
            out.append(f"{g['indent']}(FFTFusedTask0.arena.ptr() + ({g['lanes'][0]})).strided_store[width={width}]({name}, {stride})")
            count += 1
        cursor = b
    out.extend(lines[cursor:])
    assert count, 'no scalar store groups found'
    return '\n'.join(out), count


NEAR0, NEAR1, MID, FAR0, FAR12, POST = ('node_1_stage_0', 'node_1_stage_1', 'node_2_transpose',
                                         'node_3_stage_0', 'node_3_stage_1_2', 'node_4_transpose')


def strided_phase(src, name):
    body, _ = strided_rewrite(get_function(src, name))
    return replace_function(src, name, body)
