"""Complex SIMD codelet helpers (constant twiddle multiply, radix-4).

Ported unchanged from implementations/all_lengths_20261031/transforms/far_radix16.py.
"""
from __future__ import annotations

import math


def _c(value):
    text = f'{value:.9g}'
    return '0' if text in ('0', '-0') else text


def _cmul_const(z, k, n, out):
    """Lines computing out = z * W_n^k for SIMD names z=(r, i)."""
    k %= n
    zr, zi = z
    if k == 0:
        return [f'var {out}r = {zr}', f'var {out}i = {zi}']
    if 4 * k == n:  # -i
        return [f'var {out}r = {zi}', f'var {out}i = -{zr}']
    if 2 * k == n:
        return [f'var {out}r = -{zr}', f'var {out}i = -{zi}']
    if 4 * k == 3 * n:  # +i
        return [f'var {out}r = -{zi}', f'var {out}i = {zr}']
    c, s = math.cos(2 * math.pi * k / n), -math.sin(2 * math.pi * k / n)
    return [f'var {out}r = {zr} * Float32({_c(c)}) - {zi} * Float32({_c(s)})',
            f'var {out}i = {zr} * Float32({_c(s)}) + {zi} * Float32({_c(c)})']


def _radix4(x, p):
    """Forward radix-4 on complex names x[0..3]; returns output names y[0..3]."""
    (x0r, x0i), (x1r, x1i), (x2r, x2i), (x3r, x3i) = x
    lines = [f'var {p}a0r = {x0r} + {x2r}', f'var {p}a0i = {x0i} + {x2i}',
             f'var {p}a1r = {x0r} - {x2r}', f'var {p}a1i = {x0i} - {x2i}',
             f'var {p}b0r = {x1r} + {x3r}', f'var {p}b0i = {x1i} + {x3i}',
             f'var {p}b1r = {x1r} - {x3r}', f'var {p}b1i = {x1i} - {x3i}',
             f'var {p}y0r = {p}a0r + {p}b0r', f'var {p}y0i = {p}a0i + {p}b0i',
             f'var {p}y2r = {p}a0r - {p}b0r', f'var {p}y2i = {p}a0i - {p}b0i',
             f'var {p}y1r = {p}a1r + {p}b1i', f'var {p}y1i = {p}a1i - {p}b1r',
             f'var {p}y3r = {p}a1r - {p}b1i', f'var {p}y3i = {p}a1i + {p}b1r']
    return lines, [(f'{p}y{k}r', f'{p}y{k}i') for k in range(4)]
